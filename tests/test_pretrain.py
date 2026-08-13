import pytest
import torch

from pretrain import (
    accumulation_group_size,
    create_grad_scaler,
    optimizer_step_index,
    should_update_gradients,
)


@pytest.mark.parametrize(
    ("step", "expected_size", "should_update"),
    [
        (0, 4, False),
        (1, 4, False),
        (2, 4, False),
        (3, 4, True),
        (4, 2, False),
        (5, 2, True),
    ],
)
def test_handles_partial_final_accumulation_group(
    step: int, expected_size: int, should_update: bool
) -> None:
    assert accumulation_group_size(step, num_batches=6, accumulation_steps=4) == expected_size
    assert should_update_gradients(step, num_batches=6, accumulation_steps=4) is should_update


def test_creates_disabled_cpu_grad_scaler() -> None:
    scaler = create_grad_scaler("cpu", enabled=False)
    assert scaler.is_enabled() is False


def test_optimizer_step_advances_per_accumulation_group() -> None:
    assert [optimizer_step_index(0, step, 6, 4) for step in range(6)] == [0, 0, 0, 0, 1, 1]
    assert optimizer_step_index(1, 0, 6, 4) == 2
