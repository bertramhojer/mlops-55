import typing
from pathlib import Path

import datasets
import torch
import typer
from torch.utils.data import Dataset

from project.mmlu_loader import load_mmlu_dataset
from project.mmlu_processor import MMLUPreprocessor

Mode = typing.Literal["binary", "multiclass"]


class MMLUDataset(Dataset):
    """Custom Dataset class for MMLU data."""

    def __init__(self, dataset: datasets.Dataset, split: str, mode: str = "binary"):
        """Initialize MMLU Dataset.

        Args:
            dataset: Preprocessed HuggingFace dataset
            split: Dataset split ('train', 'validation', or 'test')
            mode: Either 'binary' or 'multiclass'
        """
        self.dataset = dataset[split]  # Access specific split
        self.mode = mode
        self.split = split

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
    def from_file(cls, filepath: str | Path, split: str, mode: str = "binary") -> "MMLUDataset":
        """Load dataset from processed file.

        Args:
            filepath: Path to processed dataset file
            split: Dataset split to load ('train', 'validation', or 'test')
            mode: Either 'binary' or 'multiclass'

        Returns:
            MMLUDataset instance
        """
        dataset = datasets.load_from_disk(filepath)
        return cls(dataset, split=split, mode=mode)


def get_processed_datasets(
    subjects: list[str] | None = None,
    source_split: str = "auxiliary_train",
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    mode: str = "binary",
    save_path: str | Path | None = None,
    random_seed: int = 42,
) -> dict[str, MMLUDataset]:
    """Load and preprocess MMLU dataset in specified format with splits."""
    # Load the raw dataset
    total_size = train_size + val_size + test_size
    raw_dataset = load_mmlu_dataset(subjects=subjects, split=source_split, subset_size=total_size)

    # Split the raw dataset BEFORE preprocessing
    raw_dataset = raw_dataset.shuffle(seed=random_seed)
    raw_splits = raw_dataset.train_test_split(
        train_size=train_size,
        test_size=val_size + test_size,
        shuffle=False,
    )
    raw_test_splits = raw_splits["test"].train_test_split(
        train_size=val_size,
        test_size=test_size,
        shuffle=False,
    )

    # Preprocess each split separately
    preprocessor = MMLUPreprocessor(mode=mode)
    processed_splits = {
        "train": preprocessor.preprocess_dataset(raw_splits["train"]),
        "validation": preprocessor.preprocess_dataset(raw_test_splits["train"]),
        "test": preprocessor.preprocess_dataset(raw_test_splits["test"]),
    }
    combined_dataset = datasets.DatasetDict(processed_splits)

    # Add metadata as features/columns instead of using info
    metadata = {
        "subjects": str(subjects),  # Convert to string for compatibility
        "source_split": source_split,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "mode": mode,
        "random_seed": random_seed,
    }

    # Add metadata as a new feature to each split
    for split_name, split_dataset in combined_dataset.items():
        combined_dataset[split_name] = split_dataset.add_column("metadata", [metadata] * len(split_dataset))

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined_dataset.save_to_disk(save_path)
        print(f"Saved processed dataset to {save_path}")

    # Return dictionary of dataset splits
    return {split: MMLUDataset(combined_dataset, split=split, mode=mode) for split in ["train", "validation", "test"]}


def dataset_statistics(dataset: MMLUDataset) -> None:
    """Print dataset statistics."""
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset labels: {dataset.dataset['labels'].unique()}")


def main(
    subjects: list[str] = typer.Option(None, help="List of MMLU subjects to load"),
    source_split: str = typer.Option(
        "auxiliary_train", help="Source split to use ('auxiliary_train', 'dev', 'test', or 'validation')"
    ),
    train_size: int = typer.Option(800, help="Number of training examples"),
    val_size: int = typer.Option(100, help="Number of validation examples"),
    test_size: int = typer.Option(100, help="Number of test examples"),
    mode: str = typer.Option("binary", help="Format to process data in - either 'binary' or 'multiclass'"),
    load_path: str = typer.Option(None, help="Optional path to load existing processed dataset from"),
    random_seed: int = typer.Option(42, help="Random seed for reproducible splitting"),
) -> None:
    """CLI interface for processing MMLU datasets."""
    if load_path:
        dataset = MMLUDataset.from_file(load_path, split="train", mode=mode)  # Default to train split
        print(f"Loaded dataset with {len(dataset)} examples from {load_path}")
    else:
        datasets = get_processed_datasets(
            subjects=subjects,
            source_split=source_split,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            mode=mode,
            save_path=f"data/processed/{source_split}_{mode}_splits.dataset",
            random_seed=random_seed,
        )
        for split_name, dataset in datasets.items():
            print(f"{split_name.capitalize()} split size: {len(dataset)}")


if __name__ == "__main__":
    typer.run(main)
