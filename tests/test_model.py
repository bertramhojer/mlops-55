import typing

import pytest
import pytest_check as check
import torch
from torch.optim import SGD, Adam
from transformers.modeling_outputs import SequenceClassifierOutput

from project.model import ModernBERTQA

Optimizer = type[torch.optim.Optimizer]
BATCH_SIZE = 4
BINARY_CLS = 2
SEQ_LEN = 16


@pytest.fixture(scope="module")
def model_name():
    """Fixture to provide the model name for testing."""
    return "prajjwal1/bert-tiny"


@pytest.mark.parametrize(
    ("optimizer_cls", "optimizer_params"),
    [
        # Binary classification with Adam
        (Adam, {"lr": 0.001}),
        # Binary classification with SGD
        (SGD, {"lr": 0.01, "momentum": 0.9}),
    ],
)
def test_post_forward_shape(optimizer_cls: Optimizer, optimizer_params: dict[str, typing.Any], model_name: str) -> None:
    """Test the forward pass with various configurations."""
    model = ModernBERTQA(model_name, optimizer_cls, optimizer_params)
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
    labels = torch.randint(0, BINARY_CLS, (BATCH_SIZE,))
    attention_mask = torch.ones_like(input_ids)
    output: SequenceClassifierOutput = typing.cast(
        SequenceClassifierOutput, model.forward(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    )
    check.is_instance(output.logits, torch.Tensor, "Output should be a tensor")
    check.equal(
        output.logits.shape,
        (BATCH_SIZE, BINARY_CLS),
        f"Expected logits shape {BATCH_SIZE}, got {output.logits.shape}",
    )


@pytest.mark.parametrize(
    ("optimizer_cls", "optimizer_params", "batch"),
    [
        # Binary classification example
        (
            Adam,
            {"lr": 0.001},
            {
                "input_ids": torch.randint(0, 1000, (4, 16)),
                "attention_mask": torch.ones((4, 16)),
                "labels": torch.randint(0, BINARY_CLS, (4,)),
            },
        ),
    ],
)
def test_positive_loss(
    optimizer_cls: Optimizer,
    optimizer_params: dict[str, typing.Any],
    batch: dict[str, torch.Tensor],
    model_name: str,
) -> None:
    """Test the training step with various configurations."""
    model = ModernBERTQA(model_name, optimizer_cls, optimizer_params)

    loss = model.training_step(batch, batch_idx=0)
    check.greater(loss.item(), 0, f"Loss should be positive, got {loss.item()}")


@pytest.mark.parametrize(
    ("optimizer_cls", "optimizer_params", "should_raise"),
    [
        # Valid configurations
        (Adam, {"lr": 0.001}, False),
        (SGD, {"lr": 0.01, "momentum": 0.9}, False),
        # Invalid configurations
        (Adam, {"invalid_param": 0.001}, True),
        (SGD, {"lr": 0.01, "invalid_param": 0.9}, True),
    ],
)
def test_optimizer_parameters(
    optimizer_cls: Optimizer, optimizer_params: dict[str, typing.Any], should_raise: bool, model_name: str
) -> None:
    """Test the _validate_optimizer method."""
    if should_raise:
        with pytest.raises(ValueError, match="Optimizer parameters are not compatible with optimizer class:"):
            ModernBERTQA(model_name, optimizer_cls, optimizer_params)
    else:
        model = ModernBERTQA(model_name, optimizer_cls, optimizer_params)
        check.is_instance(model.configure_optimizers(), optimizer_cls)
