from pathlib import Path

import torch

from infer import infer_model_args, load_model, load_state_dict
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from model import ModelArgs, Transformer


def tiny_model() -> Transformer:
    return Transformer(
        ModelArgs(
            dim=64,
            n_layers=2,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=128,
            multiple_of=32,
            max_seq_len=32,
        )
    )


def test_infers_checkpoint_shape() -> None:
    args = infer_model_args(tiny_model().state_dict(), n_heads=8, max_seq_len=32)
    assert args.dim == 64
    assert args.n_layers == 2
    assert args.n_heads == 8
    assert args.n_kv_heads == 4
    assert args.vocab_size == 128
    assert args.max_seq_len == 32


def test_loads_historical_checkpoint(tmp_path: Path) -> None:
    state_dict = {f"_orig_mod.{key}": value for key, value in tiny_model().state_dict().items()}
    state_dict["_orig_mod.layers.0.attention.mask"] = torch.zeros(1, 1, 32, 32)
    checkpoint_path = tmp_path / "historical.pth"
    torch.save(state_dict, checkpoint_path)

    normalized, ignored_masks = load_state_dict(checkpoint_path)
    assert ignored_masks == 1
    assert all(not key.startswith("_orig_mod.") for key in normalized)

    model, args, ignored_masks = load_model(
        checkpoint_path,
        device="cpu",
        n_heads=8,
        max_seq_len=32,
    )
    assert isinstance(model, Transformer)
    assert args.n_layers == 2
    assert ignored_masks == 1


def test_tokenizer_supports_current_transformers() -> None:
    tokenizer_path = Path(__file__).parents[1] / "chatglm_tokenizer" / "tokenizer.model"
    tokenizer = ChatGLMTokenizer(vocab_file=str(tokenizer_path))
    token_ids = tokenizer.encode("今天天气不错", add_special_tokens=False)
    assert tokenizer.decode(token_ids) == "今天天气不错"
