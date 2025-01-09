import torch
from torch.utils.data import DataLoader

from project.data import MMLUDataset, get_processed_datasets


def test_dataset_creation():
    """Test creating dataset."""
    dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary")
    assert len(dataset) > 0  # noqa: S101

    # Test getting item
    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)  # noqa: S101
    assert isinstance(item["attention_mask"], torch.Tensor)  # noqa: S101
    assert isinstance(item["labels"], torch.Tensor)  # noqa: S101

    # Test correct label dtype
    assert item["labels"].dtype == torch.float32  # noqa: S101


def test_dataloader_compatibility():
    """Test dataset works with DataLoader."""
    dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Get first batch
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 2  # noqa: S101
    assert batch["labels"].shape[0] == 2  # noqa: S101


def test_save_and_load(tmp_path):
    """Test saving and loading dataset."""
    # Create and save dataset
    save_path = tmp_path / "test.dataset"
    original_dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary", save_path=save_path)

    # Load saved dataset
    loaded_dataset = MMLUDataset.from_file(save_path, mode="binary")

    # Compare items
    assert len(original_dataset) == len(loaded_dataset)  # noqa: S101
    assert all(  # noqa: S101
        torch.equal(original_dataset[0][k], loaded_dataset[0][k]) for k in ["input_ids", "attention_mask", "labels"]
    )
