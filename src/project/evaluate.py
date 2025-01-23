import json
import pathlib
from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import pydantic_settings
import seaborn as sns
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import hydra
from project.collate import collate_fn
from project.configs import DatasetConfig, TestConfig
from project.data import load_from_dvc
from project.model import ModernBERTQA
from project.settings import settings
from project.tools import hydra_to_pydantic, pprint_config
from project.utils import aggregate_over_options

if TYPE_CHECKING:
    import datasets

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class StoreTestPreds(Callback):
    """Callback to store test predictions and labels."""

    def __init__(self):
        self.test_logits = torch.tensor([])
        self.test_labels = torch.tensor([])

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Store test logits and labels."""
        self.test_logits = torch.cat([self.test_logits, batch["logits"].cpu()])
        self.test_labels = torch.cat([self.test_labels, batch["labels"].cpu()])


class EvaluateConfig(pydantic_settings.BaseSettings):
    """Configuration for running evaluations."""

    project_name: str = pydantic.Field(..., description="Name of project")
    datamodule: DatasetConfig = pydantic.Field(..., description="Dataset configuration")
    test: TestConfig = pydantic.Field(..., description="Training configuration")

    model_config = pydantic_settings.SettingsConfigDict(cli_parse_args=True, frozen=True, arbitrary_types_allowed=True)


@hydra.main(version_base=None, config_path=str(settings.PROJECT_DIR / "configs"), config_name="test_config")
def run(cfg: DictConfig) -> None:
    """Run evaluate."""
    config: EvaluateConfig = hydra_to_pydantic(cfg, EvaluateConfig)
    pprint_config(cfg)
    run_test(config)


if torch.cuda.is_available() and torch.version.cuda.split(".")[0] == "11":
    # Will enable run on certain servers, do no delete
    import torch._dynamo  # noqa: F401

    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # noqa: SLF001


def run_test(config: EvaluateConfig):
    """Evaluate model on test set."""
    # Load processed datasets
    logger.info("Loading datasets...")
    dataset, _ = load_from_dvc(config.datamodule.data_path)
    test_dataset: datasets.Dataset = dataset["test"]
    if config.test.n_test_samples:
        test_samples_divisible = config.test.n_test_samples + 4 - (config.test.n_test_samples % 4)
        test_dataset = test_dataset.select(range(test_samples_divisible))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test.batch_size, shuffle=False, collate_fn=collate_fn
    )

    with open(f"{config.test.checkpoint_dir}/metadata.json") as f:
        metadata_dict = json.load(f)
        wandb_id = metadata_dict["wandb_run_id"]
        best_model_filename = metadata_dict["best_model_file"]

    # Load pretrained model from models and evaluate
    model = ModernBERTQA.load_from_checkpoint(best_model_filename)

    wandb_logger = WandbLogger(log_model=False, save_dir=config.test.checkpoint_dir, id=wandb_id)

    storage_callback = StoreTestPreds()

    # Evaluate model
    trainer = Trainer(
        accelerator="gpu" if DEVICE.type == "cuda" else None,
        devices=list(range(torch.cuda.device_count())),
        callbacks=[storage_callback],
        logger=wandb_logger,
    )
    results = trainer.test(model, dataloaders=test_loader)
    all_preds, all_labels = storage_callback.test_logits, storage_callback.test_labels

    all_preds, all_labels = aggregate_over_options(all_preds, all_labels, num_options=4)

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = accuracy_score(all_labels, all_preds)

    # Check for label biases
    classes = np.unique(all_labels)
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)
    true_label_distribution = {cls: label_counts[cls] / len(all_labels) for cls in classes}
    predicted_label_distribution = {cls: pred_counts[cls] / len(all_preds) for cls in classes}
    label_biases = {
        int(cls): abs(predicted_label_distribution.get(cls, 0) - true_label_distribution.get(cls, 0)) for cls in classes
    }

    # Make confusion matrix sklearn and save to dir
    confusion_matrix_results = confusion_matrix(all_labels, all_preds)
    confusion_matrix_path = pathlib.Path(config.test.output_dir) / "confusion_matrix.png"
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix_results, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path)
    logger.info(f"Confusion matrix saved to: {confusion_matrix_path}")

    # Save evaluation results
    output_path = pathlib.Path(config.test.output_dir) / "evaluation_results.json"
    with open(output_path, "w") as f:
        results_dict = {"results": results, "f1": f1, "accuracy": accuracy, "label_biases": label_biases}
        json.dump(results_dict, f, indent=4)
    logger.info(f"Evaluation results saved to output_path: {output_path}")

    label_biases_table = pd.DataFrame(label_biases.items(), columns=["Label", "Bias"])
    results_table = pd.DataFrame(
        {"Metric": ["F1", "Accuracy", "Test Loss"], "Value": [f1, accuracy, results[0]["test/loss"]]}
    )

    wandb_logger.log_table("test/metrics", dataframe=results_table)
    wandb_logger.log_table("test/label_biases", dataframe=label_biases_table)
    wandb_logger.log_image("test/confusion_matrix", [str(confusion_matrix_path)], caption=["Confusion Matrix"])


if __name__ == "__main__":
    run()
