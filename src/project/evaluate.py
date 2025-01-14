import json
import pathlib
from collections import Counter

import hydra
import pydantic
import pydantic_settings
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
from omegaconf import DictConfig

from project.configs import DatasetConfig, TestConfig
from project.data import get_processed_datasets
from project.model import ModernBERTQA
from project.tools import hydra_to_pydantic, pprint_config

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class TestConfig(pydantic_settings.BaseSettings):
    """Configuration for running experiements."""

    project_name: str = pydantic.Field(..., description="Name of project")
    datamodule: DatasetConfig = pydantic.Field(..., description="Dataset configuration")
    test: TestConfig = pydantic.Field(..., description="Testing configuration")

    model_config = pydantic_settings.SettingsConfigDict(cli_parse_args=True, frozen=True, arbitrary_types_allowed=True)



class StoreTestPreds(Callback):
    """Callback to store test predictions and labels."""
    def __init__(self):
        self.test_logits = []
        self.test_labels = []

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Store test logits and labels."""
        self.test_logits.extend(batch["logits"].argmax(dim=-1).cpu().numpy())
        self.test_labels.extend(batch["labels"].cpu().numpy())

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="test_config")
def run(cfg: DictConfig) -> None:
    """Run training loop."""
    config: TestConfig = hydra_to_pydantic(cfg, TestConfig)
    pprint_config(cfg)
    run_test(config)

if (
    torch.cuda.is_available()
    and torch.version.cuda.split(".")[0] == "11"
):
    # Will enable run on certain servers, do no delete
    import torch._dynamo  # noqa: F401

    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # noqa: SLF001


def run_test(config: TestConfig):
    """
    Train model, saves model to output_dir.
    """
    # Load processed datasets
    logger.info("Loading datasets...")
    test_dataset = get_processed_datasets(
        source_split="auxiliary_train",  
        subjects=config.datamodule.subjects,
        mode=config.datamodule.mode,
        train_size=config.datamodule.train_subset_size,
        val_size=config.datamodule.val_subset_size,
        test_size=config.datamodule.test_subset_size,
    )["test"]
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False)


    # Load pretrained model from models and evaluate
    model = ModernBERTQA.load_from_checkpoint(config.test.checkpoint_path)

    storage_callback = StoreTestPreds()

    # Evaluate model
    trainer = Trainer(
        accelerator="gpu" if DEVICE.type == "cuda" else None,
        devices=list(range(torch.cuda.device_count())),
        callbacks=[storage_callback]
    )
    results = trainer.test(model, dataloaders=test_loader)
    all_preds, all_labels = storage_callback.test_logits, storage_callback.test_labels

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = accuracy_score(all_labels, all_preds)

    # Check for label biases
    class_counts = len(np.unique(all_labels))
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)
    true_label_distribution = {cls: label_counts[cls] / len(all_labels) for cls in range(class_counts)}
    predicted_label_distribution = {cls: pred_counts[cls] / len(all_preds) for cls in range(class_counts)}
    label_biases = {
        cls: abs(predicted_label_distribution.get(cls, 0)
                 - true_label_distribution.get(cls, 0)) for cls in range(class_counts)
    }

    # Save evaluation results
    output_path = pathlib.Path(config.test.output_dir) / "evaluation_results.json"
    with open(output_path, "w") as f:
        results_dict = {
            "results": results,
            "f1": f1,
            "accuracy": accuracy,
            "label_biases": label_biases
        }
        json.dump(results_dict, f, indent=4)
    logger.info(f"Evaluation results saved to output_path: {output_path}")


if __name__ == "__main__":
    run()
