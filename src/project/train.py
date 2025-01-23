import json
import typing
from typing import TYPE_CHECKING

import pydantic
import pydantic_settings
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig

import hydra
from project.collate import collate_fn
from project.configs import DatasetConfig, OptimizerConfig, TrainConfig
from project.data import load_from_dvc
from project.model import ModernBERTQA
from project.settings import settings
from project.tools import hydra_to_pydantic, pprint_config

if TYPE_CHECKING:
    import datasets


class ExperimentConfig(pydantic_settings.BaseSettings):
    """Configuration for running experiements."""

    project_name: str = pydantic.Field(..., description="Name of project")
    datamodule: DatasetConfig = pydantic.Field(..., description="Dataset configuration")
    optimizer: OptimizerConfig = pydantic.Field(..., description="Optimizer configuration")
    train: TrainConfig = pydantic.Field(..., description="Training configuration")

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", frozen=True, arbitrary_types_allowed=True)


@hydra.main(version_base="1.3", config_path=str(settings.PROJECT_DIR / "configs"), config_name="train_config")
def run(cfg: DictConfig) -> None:
    """Run training loop."""
    config: ExperimentConfig = hydra_to_pydantic(cfg, ExperimentConfig)
    pprint_config(cfg)
    run_train(config)


if torch.cuda.is_available() and torch.version.cuda.split(".")[0] == "11":  # type: ignore  # noqa: PGH003
    # Will enable run on certain servers, do no delete
    import torch._dynamo  # noqa: F401

    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # noqa: SLF001


def run_train(config: ExperimentConfig):
    """Train model, saves model to output_dir."""
    train_output_dir = str(settings.PROJECT_DIR / config.train.output_dir)

    wandb_logger = WandbLogger(
        project=settings.WANDB_PROJECT, entity=settings.WANDB_ENTITY, log_model=True
    )

    # Load processed datasets
    logger.info(f"Loading datasets from {config.datamodule.file_name}...")

    dataset, _ = load_from_dvc(config.datamodule.file_name)
    train_dataset: datasets.Dataset = dataset["train"]
    val_dataset: datasets.Dataset = dataset["validation"]

    if config.train.n_train_samples:
        train_dataset = train_dataset.select(range(config.train.n_train_samples))
    if config.train.n_val_samples:
        val_dataset = val_dataset.select(range(config.train.n_val_samples))

    train_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, val_dataset),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize model
    model = ModernBERTQA(
        config.train.model_name,
        optimizer_cls=getattr(torch.optim, config.optimizer.optimizer_name),
        optimizer_params=config.optimizer.optimizer_params,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_output_dir,
        monitor=config.train.monitor,
        mode=config.train.mode,
        filename="model-{epoch:02d}-{" + config.train.monitor + ":.2f}",
        save_top_k=1,  # Saves the best model only
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.train.monitor, patience=config.train.patience, verbose=True, mode=config.train.mode
    )

    # Train and save model
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=str(settings.DEVICE),
        max_epochs=config.train.epochs,
        devices="auto",
        default_root_dir=train_output_dir,
        logger=wandb_logger,
        log_every_n_steps=5,
        precision="32",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    with open(f"{train_output_dir}/metadata.json", "w") as f:
        metadata = {"best_model_file": checkpoint_callback.best_model_path, "wandb_run_id": wandb_logger.experiment.id}
        json.dump(metadata, f)


if __name__ == "__main__":
    run()
