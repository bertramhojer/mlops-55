from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset

from project.mmlu_loader import load_mmlu_dataset
from project.mmlu_processor import MMLUPreprocessor


class MMLUDataset(Dataset):
    """Custom Dataset class for MMLU data."""

    def __init__(self, dataset: datasets.Dataset, mode: str = "binary"):
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
    def from_file(cls, filepath: str | Path, mode: str = "binary") -> "MMLUDataset":
        """Load dataset from processed file.

        Args:
            filepath: Path to processed dataset file
            mode: Either 'binary' or 'multiclass'

        Returns:
            MMLUDataset instance
        """
        dataset = datasets.load_from_disk(filepath)
        return cls(dataset, mode=mode)


def get_processed_datasets(
    subjects: list[str] | None = None,
    split: str = "test",
    subset_size: int = 100,
    mode: str = "binary",
    save_path: str | Path | None = None,
) -> MMLUDataset:
    """Load and preprocess MMLU dataset in specified format.

    Args:
        subjects: List of MMLU subjects to load
        split: Dataset split ('train', 'test', or 'validation')
        subset_size: Number of examples to load
        mode: Format to process data in - either 'binary' or 'multiclass'
        save_path: Optional path to save processed dataset to

    Returns:
        MMLUDataset instance ready for training
    """
    # First load the dataset
    raw_dataset = load_mmlu_dataset(subjects=subjects, split=split, subset_size=subset_size)

    # Process in specified format
    preprocessor = MMLUPreprocessor(mode=mode)
    processed_dataset = preprocessor.preprocess_dataset(raw_dataset)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.save_to_disk(save_path)
        print(f"Saved processed dataset to {save_path}")

    return MMLUDataset(processed_dataset, mode=mode)


if __name__ == "__main__":
    # Example usage
    dataset = get_processed_datasets(
        subjects=None, split="test", subset_size=100, mode="binary", save_path="data/processed/test_binary_n100.dataset"
    )
    print(f"Created dataset with {len(dataset)} examples")
