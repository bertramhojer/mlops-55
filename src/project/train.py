import pathlib

import hydra
import pydantic
import pydantic_settings
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig
from dotenv import load_dotenv

from project.configs import DatasetConfig, OptimizerConfig, TrainConfig
from project.data import get_processed_datasets
from project.model import ModernBERTQA
from project.tools import hydra_to_pydantic, pprint_config

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Initialize wandb logger
load_dotenv()

class ExperimentConfig(pydantic_settings.BaseSettings):
    """Configuration for running experiements."""

    project_name: str = pydantic.Field(..., description="Name of project")
    datamodule: DatasetConfig = pydantic.Field(..., description="Dataset configuration")
    optimizer: OptimizerConfig = pydantic.Field(..., description="Optimizer configuration")
    train: TrainConfig = pydantic.Field(..., description="Training configuration")

    model_config = pydantic_settings.SettingsConfigDict(cli_parse_args=True, frozen=True, arbitrary_types_allowed=True)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train_config")
def run(cfg: DictConfig) -> None:
    """Run training loop."""
    config: ExperimentConfig = hydra_to_pydantic(cfg, ExperimentConfig)
    pprint_config(cfg)
    run_train(config)


if (
    torch.cuda.is_available()
    and torch.version.cuda.split(".")[0] == "11"
):
    # Will enable run on certain servers, do no delete
    import torch._dynamo  # noqa: F401

    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # noqa: SLF001


def run_train(config: ExperimentConfig):
    """Train model, saves model to output_dir.

    TODO: fix binary classification.
    """

    train_output_dir = str(PROJECT_ROOT / config.train.output_dir)

    wandb_logger = WandbLogger(log_model=False, save_dir=train_output_dir)

    # Load processed datasets
    logger.info("Loading datasets...")
    datasets = get_processed_datasets(
        source_split="auxiliary_train",  
        subjects=config.datamodule.subjects,
        mode=config.datamodule.mode,
        train_size=config.datamodule.train_subset_size,
        val_size=config.datamodule.val_subset_size,
        test_size=config.datamodule.test_subset_size,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    num_choices = train_dataset.__getoptions__()

    # Initialize model
    model = ModernBERTQA(
        config.train.model_name,
        num_choices=num_choices,
        optimizer_cls=getattr(torch.optim, config.optimizer.optimizer_name),
        optimizer_params=config.optimizer.optimizer_params,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_output_dir, monitor=config.train.monitor, mode=config.train.mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.train.monitor, patience=config.train.patience, verbose=True, mode=config.train.mode
    )

    # Train and save model
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if DEVICE.type == "cuda" else "cpu",
        max_epochs=config.train.epochs,
        devices=list(range(torch.cuda.device_count())),
        default_root_dir=train_output_dir,
        logger=wandb_logger,
        log_every_n_steps=5,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    run_id = wandb_logger.experiment.id
    with open(f"{train_output_dir}/wandb_id.txt", "w") as f:
        f.write(run_id)



if __name__ == "__main__":
    run()
