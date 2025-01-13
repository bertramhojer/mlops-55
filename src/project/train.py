import pathlib
import typing

import hydra
import pydantic_settings
import torch
import typer
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from omegaconf import DictConfig, OmegaConf

from project.configs import DatasetConfig, OptimizerConfig, TrainConfig
from project.data import MMLUDataset
from project.tools import pprint_config

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

train_app = typer.Typer()


class ExperimentConfig(pydantic_settings.BaseSettings):
    """Configuration for running experiements."""

    project_name: str = "MMLU Classification"
    datamodule: DatasetConfig
    optimizer: OptimizerConfig
    train: TrainConfig

    model_config = pydantic_settings.SettingsConfigDict(cli_parse_args=True, frozen=True, arbitrary_types_allowed=True)


# Create Hydra-compatible dataclass that describes `ExperimentConfig`
HydraConf = builds(ExperimentConfig, populate_full_signature=True)

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=HydraConf)


def hydra_to_pydantic(config: DictConfig) -> ExperimentConfig:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config_dict = typing.cast(dict[str, typing.Any], OmegaConf.to_object(config))
    return ExperimentConfig(**config_dict)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def run(cfg: DictConfig) -> None:
    """Run training loop."""
    config: ExperimentConfig = hydra_to_pydantic(cfg)
    pprint_config(cfg)
    train(config)


@train_app.command("train")
def train(config: ExperimentConfig) -> None:
    """Training loop."""
    dataset = MMLUDataset.from_file(config.datamodule.path_to_data)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train.eval_batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train.eval_batch_size)


if __name__ == "__main__":
    run()
