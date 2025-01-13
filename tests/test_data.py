from pathlib import Path

from project.data import MMLUDataset, get_processed_datasets


def test_get_processed_datasets_with_save():
    """Test dataset processing with save functionality."""
    # Create a temporary save path
    save_path = Path("tests/temp_dataset")

    # Get dataset with save
    dataset = get_processed_datasets(
        subjects=["philosophy"], split="test", subset_size=5, mode="binary", save_path=save_path
    )

    assert isinstance(dataset, MMLUDataset)  # noqa: S101
    assert save_path.exists()  # noqa: S101

    # Load the saved dataset and verify
    loaded_dataset = MMLUDataset.from_file(save_path, mode="binary")
    assert len(loaded_dataset) == len(dataset)  # noqa: S101

    # Cleanup
    import shutil

    shutil.rmtree(save_path)


def test_dataset_metadata():
    """Test that dataset metadata is properly stored."""
    dataset = get_processed_datasets(subjects=["philosophy"], split="test", subset_size=5, mode="binary")

    # Check that the dataset info contains our metadata
    assert "Processed MMLU dataset" in dataset.dataset.info.description  # noqa: S101
    assert "subjects: ['philosophy']" in dataset.dataset.info.description  # noqa: S101
    assert "split: test" in dataset.dataset.info.description  # noqa: S101
