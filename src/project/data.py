import os
import typing
from pathlib import Path

import datasets
import torch
import typer
from dvc.repo import Repo
from torch.utils.data import Dataset

from project.mmlu_loader import load_mmlu_dataset
from project.mmlu_processor import MMLUPreprocessor

Mode = typing.Literal["binary", "multiclass"]


class MMLUDataset(Dataset):
    """Custom Dataset class for MMLU data."""

    def __init__(self, dataset: datasets.Dataset, mode: Mode = "binary"):
        """Initialize MMLU Dataset.

        Args:
            dataset: Preprocessed HuggingFace dataset
            mode: Either 'binary' or 'multiclass'
        """
        self.dataset = dataset
        self.mode = mode

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getoptions__(self) -> int:
        """Get unique labels in dataset."""
        return len(set(self.dataset["labels"]))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Index of item to get

        Returns:
            Dictionary containing input_ids, attention_mask and labels tensors
        """
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"], dtype=torch.float32 if self.mode == "binary" else torch.long),
        }

    @classmethod
    def from_file(cls, filepath: str, from_dvc: bool = True) -> "MMLUDataset":
        """Load dataset from processed file.

        Args:
            filepath: Path to processed dataset file (local or remote)
            from_dvc: Whether the file is stored in a remote DVC storage
        Returns:
            MMLUDataset instance
        """
        if from_dvc:
            dataset = datasets.load_from_disk(filepath, storage_options={"project": "mmlu-bucket"})
        else:
            dataset = datasets.load_from_disk(filepath)
        return cls(dataset)  # type: ignore[arg-type]


def get_processed_datasets(
    subjects: list[str] | None = None,
    split: str = "test",
    subset_size: int = 100,
    mode: typing.Literal["binary", "multiclass"] = "binary",
    save_path: str | Path | None = None,
    remote: str = "remote_storage",
) -> MMLUDataset:
    """Load and preprocess MMLU dataset in specified format.

    Args:
        subjects: List of MMLU subjects to load
        split: Dataset split ('train', 'test', or 'validation')
        subset_size: Number of examples to load
        mode: Format to process data in - either 'binary' or 'multiclass'
        save_path: Optional path to save processed dataset to
        remote: Name of the DVC remote to use (default: 'remote_storage')

    Returns:
        MMLUDataset instance ready for training
    """
    # First load the dataset
    raw_dataset = load_mmlu_dataset(subjects=subjects, split=split, subset_size=subset_size)

    # Process in specified format
    preprocessor = MMLUPreprocessor(mode=mode)
    processed_dataset = preprocessor.preprocess_dataset(raw_dataset)

    # Add metadata to the dataset info
    processed_dataset.info.description = f"Processed MMLU dataset ({mode} mode)"
    # Store metadata in the description field as a string
    metadata_str = f"subjects: {subjects}, split: {split}, subset_size: {subset_size}, mode: {mode}"
    processed_dataset.info.description += f"\nMetadata: {metadata_str}"

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.save_to_disk(save_path)

        # init dvc repo
        repo = Repo(".")

        # add dataset to dvc
        repo.add(str(save_path))

        # push to remote
        repo.push(remote=remote)

        print(f"Saved processed dataset to {save_path} and pushed to {remote} remote")

    return MMLUDataset(processed_dataset, mode=mode)


def main(
    subjects: list[str] = typer.Option(None, help="List of MMLU subjects to load"),
    split: str = typer.Option("test", help="Dataset split ('auxiliary_train', 'dev', 'test', or 'validation')"),
    subset_size: int = typer.Option(100, help="Number of examples to load"),
    mode: str = typer.Option("binary", help="Format to process data in - either 'binary' or 'multiclass'"),
    load_path: str = typer.Option(None, help="Optional path to load existing processed dataset from"),
    remote: str = typer.Option("remote_storage", help="Name of the DVC remote to use"),
) -> None:
    """CLI interface for processing MMLU datasets."""
    if load_path:
        # Pull form dVc if the file doesn't exist locally
        if not os.path.exists(load_path):
            repo = Repo(".")
            repo.pull(remote=remote, target=[load_path])

        dataset = MMLUDataset.from_file(load_path, mode=mode)
        print(f"Loaded dataset with {len(dataset)} examples from {load_path}")
    else:
        dataset = get_processed_datasets(
            subjects=subjects,
            split=split,
            subset_size=subset_size,
            mode=mode,
            save_path=f"data/processed/{split}_{mode}_n{subset_size}.dataset",
            remote=remote,
        )
        print(f"Created dataset with {len(dataset)} examples")


if __name__ == "__main__":
    typer.run(main)
