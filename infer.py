"""Run inference with Baby-Llama2-Chinese checkpoints.

The loader supports both current state dicts and historical checkpoints that
contain ``module.``/``_orig_mod.`` prefixes or saved causal attention masks.
"""

import argparse
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from model import ModelArgs, Transformer


def load_state_dict(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], int]:
    """Load and normalize a historical or current model state dict."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions without the weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    state_dict = {}
    ignored_masks = 0
    for key, value in checkpoint.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        if key.endswith("attention.mask"):
            ignored_masks += 1
            continue
        state_dict[key] = value
    return state_dict, ignored_masks


def infer_model_args(
    state_dict: Dict[str, torch.Tensor],
    n_heads: int = 8,
    max_seq_len: Optional[int] = None,
) -> ModelArgs:
    """Infer the model dimensions that are represented in a state dict."""
    embedding = state_dict.get("tok_embeddings.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("Checkpoint does not contain tok_embeddings.weight")

    vocab_size, dim = embedding.shape
    layer_ids = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"layers\.(\d+)\.", key))
    }
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError("Checkpoint contains an incomplete Transformer layer sequence")
    if dim % n_heads:
        raise ValueError(f"Model dimension {dim} is not divisible by {n_heads} heads")

    key_weight = state_dict.get("layers.0.attention.wk.weight")
    if key_weight is None:
        raise ValueError("Checkpoint does not contain layers.0.attention.wk.weight")
    head_dim = dim // n_heads
    if key_weight.shape[0] % head_dim:
        raise ValueError("Cannot infer the number of key/value heads")

    return ModelArgs(
        dim=dim,
        n_layers=len(layer_ids),
        n_heads=n_heads,
        n_kv_heads=key_weight.shape[0] // head_dim,
        vocab_size=vocab_size,
        multiple_of=32,
        max_seq_len=max_seq_len or (512 if dim <= 512 else 1024),
        dropout=0.0,
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(
    checkpoint_path: Path,
    device: str,
    n_heads: int = 8,
    max_seq_len: Optional[int] = None,
) -> Tuple[Transformer, ModelArgs, int]:
    state_dict, ignored_masks = load_state_dict(checkpoint_path)
    model_args = infer_model_args(state_dict, n_heads=n_heads, max_seq_len=max_seq_len)
    model = Transformer(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, model_args, ignored_masks


@torch.inference_mode()
def generate(
    model: Transformer,
    tokenizer: ChatGLMTokenizer,
    prompt: str,
    mode: str,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if mode == "sft":
        token_ids.append(tokenizer.special_tokens["<bos>"])
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    use_autocast = device.startswith("cuda") and dtype != torch.float32
    context = torch.amp.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()
    with context:
        output = model.generate(
            tokens,
            eos=tokenizer.special_tokens["<eos>"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    generated_ids = output[0, len(token_ids) :].tolist()
    return tokenizer.decode(generated_ids).strip()


def prompts_from_args(prompts: Iterable[str]) -> Iterable[str]:
    prompts = list(prompts)
    if prompts:
        yield from prompts
        return
    while True:
        try:
            prompt = input("Prompt (Ctrl-D to exit): ").strip()
        except EOFError:
            return
        if prompt:
            yield prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("pretrain", "sft"), default="sft")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("chatglm_tokenizer/tokenizer.model"),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)

    model, model_args, ignored_masks = load_model(
        args.checkpoint,
        device=device,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
    )
    if args.compile:
        model = torch.compile(model)
    tokenizer = ChatGLMTokenizer(vocab_file=str(args.tokenizer))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Loaded {args.checkpoint} on {device} ({dtype}); "
        f"dim={model_args.dim}, layers={model_args.n_layers}, "
        f"heads={model_args.n_heads}, parameters={parameter_count:,}, "
        f"ignored_legacy_masks={ignored_masks}"
    )
    for prompt in prompts_from_args(args.prompt):
        answer = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            mode=args.mode,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
        )
        print(f"\nPrompt: {prompt}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
