import pathlib
import typing

import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Configuration for data module."""

    data_path: str = pydantic.Field(..., description="Path to data directory in DVC")


class OptimizerConfig(pydantic.BaseModel):
    """Configuration for optimizer."""

    optimizer_name: str = pydantic.Field(..., description="Name of optimizer")
    optimizer_params: dict[str, typing.Any] = pydantic.Field(..., description="Parameters for optimizer")


class TrainConfig(pydantic.BaseModel):
    """Configuration for training."""

    model_name: str = pydantic.Field(..., description="Model name to train")
    seed: int = pydantic.Field(..., description="Random seed for reproducibility")
    device: str = pydantic.Field(..., description="Device to use for training")
    epochs: int = pydantic.Field(..., description="Number of epochs to train")
    batch_size: int = pydantic.Field(..., description="Batch size for training")
    eval_batch_size: int = pydantic.Field(..., description="Batch size for evaluation")
    output_dir: str = pydantic.Field(..., description="Output directory for training")
    monitor: str = pydantic.Field(..., description="Metric to monitor for early stopping")
    mode: str = pydantic.Field(..., description="Mode for monitoring")
    patience: int = pydantic.Field(..., description="Patience for early stopping")
    n_train_samples: int | None = pydantic.Field(None, description="Number of training samples")
    n_val_samples: int | None = pydantic.Field(None, description="Number of validation samples")

    @pydantic.field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, v: str) -> str:
        root_dir = pathlib.Path(__file__).parent.parent.parent
        return pathlib.Path(root_dir, v).as_posix()


class TestConfig(pydantic.BaseModel):
    """Configuration for testing."""

    checkpoint_dir: str = pydantic.Field(..., description="Path to model checkpoint")
    output_dir: str = pydantic.Field(..., description="Path to save evaluation results")
    batch_size: int = pydantic.Field(..., description="Batch size for testing")
    seed: int = pydantic.Field(..., description="Random seed for reproducibility")
    device: str = pydantic.Field(..., description="Device to use for testing")
    n_test_samples: int | None = pydantic.Field(None, description="Number of test samples")
