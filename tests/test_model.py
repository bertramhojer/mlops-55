import typing

import pytest
import pytest_check as check
import torch
from torch.optim import SGD, Adam

from project.model import ModernBERTQA

Optimizer = type[torch.optim.Optimizer]
BATCH_SIZE = 4
SEQ_LEN = 16


@pytest.fixture(scope="module")
def model_name():
    """Fixture to provide the model name for testing."""
    return "prajjwal1/bert-tiny"


@pytest.mark.parametrize(
    ("num_choices", "optimizer_cls", "optimizer_params"),
    [
        # Binary classification with Adam
        (2, Adam, {"lr": 0.001}),
        # Binary classification with SGD
        (2, SGD, {"lr": 0.01, "momentum": 0.9}),
        # Multi-class classification with Adam
        (4, Adam, {"lr": 0.001}),
        # Multi-class classification with SGD
        (4, SGD, {"lr": 0.01, "momentum": 0.9}),
    ],
)
def test_post_forward_shape(
    num_choices: int, optimizer_cls: Optimizer, optimizer_params: dict[str, typing.Any], model_name: str
) -> None:
    """Test the forward pass with various configurations."""
    model = ModernBERTQA(model_name, num_choices, optimizer_cls, optimizer_params)
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, num_choices, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    logits = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    check.is_instance(logits, torch.Tensor, "Output should be a tensor")
    check.equal(
        logits.shape,
        (BATCH_SIZE, num_choices),
        f"Expected logits shape {(BATCH_SIZE, num_choices)}, got {logits.shape}",
    )


@pytest.mark.parametrize(
    ("num_choices", "optimizer_cls", "optimizer_params", "batch"),
    [
        # Binary classification example
        (
            2,
            Adam,
            {"lr": 0.001},
            {
                "input_ids": torch.randint(0, 1000, (4, 2, 16)),
                "attention_mask": torch.ones((4, 2, 16)),
                "label": torch.randint(0, 2, (4,)),
            },
        ),
        # Multi-class classification example
        (
            4,
            SGD,
            {"lr": 0.01, "momentum": 0.9},
            {
                "input_ids": torch.randint(0, 1000, (8, 4, 32)),
                "attention_mask": torch.ones((8, 4, 32)),
                "label": torch.randint(0, 4, (8,)),
            },
        ),
    ],
)
def test_positive_loss(
    num_choices: int,
    optimizer_cls: Optimizer,
    optimizer_params: dict[str, typing.Any],
    batch: dict[str, torch.Tensor],
    model_name: str,
) -> None:
    """Test the training step with various configurations."""
    model = ModernBERTQA(model_name, num_choices, optimizer_cls, optimizer_params)

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
    num_choices = 2
    if should_raise:
        with pytest.raises(ValueError, match="Optimizer parameters are not compatible with optimizer class:"):
            ModernBERTQA(model_name, num_choices, optimizer_cls, optimizer_params)
    else:
        model = ModernBERTQA(model_name, num_choices, optimizer_cls, optimizer_params)
        check.is_instance(model.configure_optimizers(), optimizer_cls)
