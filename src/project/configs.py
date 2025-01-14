import pathlib
import typing
import typing_extensions

import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Configuration for data module."""

    subjects: list[str] | None = pydantic.Field(default=None, description="Subjects to include in dataset")
    split: str = pydantic.Field(..., description="Split to use for dataset")
    mode: typing.Literal["binary", "multiclass"] = pydantic.Field(..., description="Mode for dataset")
    train_subset_size: int = pydantic.Field(..., description="Subset size for dataset")
    val_subset_size: int = pydantic.Field(..., description="Subset size for validation dataset")
    test_subset_size: int = pydantic.Field(..., description="Subset size for test dataset")

    @property
    def path_to_data(self) -> str:
        """Get path to data directory."""
        _data_path = pathlib.Path(f"data/processed/{self.split}_{self.mode}_n{self.train_subset_size}.dataset")
        return _data_path.as_posix()


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

    @pydantic.field_validator("output_dir", mode="before")
    @classmethod
    def _validate_output_dir(cls, v: str) -> str:
        root_dir = pathlib.Path(__file__).parent.parent.parent
        output_path = pathlib.Path(root_dir, v).as_posix()
        if pathlib.Path(output_path).exists():
            raise ValueError(f"Output directory already exists: {output_path}")
        return output_path

class TestConfig(pydantic.BaseModel):
    """Configuration for testing."""

    checkpoint_dir: str = pydantic.Field(..., description="Path to model checkpoint")
    output_dir: str = pydantic.Field(..., description="Path to save evaluation results")
    batch_size: int = pydantic.Field(..., description="Batch size for testing")
    seed: int = pydantic.Field(..., description="Random seed for reproducibility")
    device: str = pydantic.Field(..., description="Device to use for testing")
