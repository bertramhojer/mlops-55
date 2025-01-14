import json
from collections import Counter

import numpy as np
import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score

from project.data import get_processed_datasets
from project.model import ModernBERTQA

# Will enable run on certain servers, do no delete
#import torch._dynamo  noqa: F401
#torch._dynamo.config.suppress_errors = True  noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


class StoreTestPreds(Callback):
    """Callback to store test predictions and labels."""
    def __init__(self):
        self.test_logits = []
        self.test_labels = []

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Store test logits and labels."""
        self.test_logits.extend(batch["logits"].argmax(dim=-1).cpu().numpy())
        self.test_labels.extend(batch["labels"].cpu().numpy())


@app.command()
def test_model(
    subjects: list[str] | None = typer.Option(None, help="Subjects to include in dataset"),
    batch_size: int = typer.Option(16, help="Batch size for testing"),
    mode: str = typer.Option("multiclass", help="Mode for testing"),
    path_to_checkpoint: str = typer.Option("models/checkpoint.ckpt", help="Path to model checkpoint"),
    path_to_save: str = typer.Option("models/", help="Path to save evaluation results"),
    test_subset_size: int | None = typer.Option(None, help="Subset size for testing")
):
    """
    Train model, saves model to output_dir.

    Args:
            subjects: Subjects to include in dataset
            batch_size: Batch size for training
            mode: Mode for training
            path_to_checkpoint: Path to model checkpoint
            path_to_save: Path to save evaluation results
            test_subset_size: Subset size for testing

    Returns:
            None
    """
    # Load processed datasets
    print("Loading datasets...")
    test_dataset = get_processed_datasets(
        split="test",
        subjects=subjects,
        mode=mode,
        subset_size=test_subset_size
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Load pretrained model from models and evaluate
    model = ModernBERTQA.load_from_checkpoint(f"{path_to_checkpoint}")

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
    output_path = f"{path_to_save}evaluation_results.json"
    with open(output_path, "w") as f:
        results_dict = {
            "results": results,
            "f1": f1,
            "accuracy": accuracy,
            "label_biases": label_biases
        }
        json.dump(results_dict, f, indent=4)
    print(f"Evaluation results saved to {path_to_save}evaluation_results.json")


if __name__ == "__main__":
    app()
