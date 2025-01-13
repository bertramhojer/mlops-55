import pathlib
import typing

import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Configuration for data module."""

    split: str
    mode: typing.Literal["binary", "multi-class"]
    subset_size: int

    @property
    def path_to_data(self) -> str:
        """Get path to data directory."""
        _data_path = pathlib.Path(f"data/processed/{self.split}_{self.mode}_n{self.subset_size}.dataset")
        return _data_path.as_posix()


class OptimizerConfig(pydantic.BaseModel):
    """Configuration for optimizer."""

    optimizer_name: str
    optimizer_params: dict[str, typing.Any]


class TrainConfig(pydantic.BaseModel):
    """Configuration for training."""

    seed: int
    device: str
    epochs: int
    batch_size: int
    eval_batch_size: int
