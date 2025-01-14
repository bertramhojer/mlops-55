import pytest

from project.data import MMLUDataset, get_processed_datasets
from project.mmlu_loader import list_available_subjects, load_mmlu_dataset


def test_load_single_subject():
    """Test loading a single MMLU subject."""
    dataset = load_mmlu_dataset(subjects=["anatomy"], split="test")
    assert dataset is not None  # noqa: S101
    assert len(dataset) > 0  # noqa: S101
    assert all(col in dataset.column_names for col in ["question", "choices", "answer"])  # noqa: S101


def test_load_multiple_subjects():
    """Test loading multiple MMLU subjects."""
    dataset = load_mmlu_dataset(subjects=["anatomy", "philosophy"], split="test")
    assert dataset is not None  # noqa: S101
    assert len(dataset) > 0  # noqa: S101


def test_subset_size():
    """Test loading a subset of the data."""
    subset_size = 10
    dataset = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=subset_size)
    assert len(dataset) == subset_size  # noqa: S101


def test_reproducibility():
    """Test that setting seed produces same subset."""
    subset_size = 5
    seed = 42

    dataset1 = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=subset_size, random_seed=seed)
    dataset2 = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=subset_size, random_seed=seed)

    assert all(d1 == d2 for d1, d2 in zip(dataset1, dataset2))  # noqa: S101


def test_error_empty_subjects():
    """Test error handling for empty subjects list."""
    with pytest.raises(ValueError, match="Must provide at least one subject"):
        load_mmlu_dataset(subjects=[])


def test_error_invalid_subject():
    """Test error handling for invalid subject name."""
    with pytest.raises(RuntimeError):
        load_mmlu_dataset(subjects=["not_a_real_subject"])


def test_error_negative_subset():
    """Test error handling for negative subset size."""
    with pytest.raises(ValueError, match="subset_size must be positive"):
        load_mmlu_dataset(subjects=["anatomy"], subset_size=-1)


def test_list_subjects():
    """Test listing available subjects."""
    subjects = list_available_subjects()
    assert len(subjects) > 0  # noqa: S101
    assert "anatomy" in subjects  # noqa: S101
    assert "philosophy" in subjects  # noqa: S101
    assert all(isinstance(subject, str) for subject in subjects)  # noqa: S101


def test_get_processed_datasets_with_save(tmp_path):
    """Test dataset processing with save functionality."""
    from pathlib import Path

    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    save_path = test_dir / "test_dataset"
    dvc_file = Path(str(save_path) + ".dvc")

    try:
        dataset = get_processed_datasets(
            subjects=["philosophy"], split="test", subset_size=5, mode="binary", save_path=save_path
        )

        assert isinstance(dataset, MMLUDataset)  # noqa: S101
        assert save_path.exists()  # noqa: S101

        loaded_dataset = MMLUDataset.from_file(save_path)
        assert len(loaded_dataset) == len(dataset)  # noqa: S101
    finally:
        # Clean up
        import shutil

        if save_path.exists():
            shutil.rmtree(save_path)
        if dvc_file.exists():
            dvc_file.unlink()
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_dataset_metadata():
    """Test that dataset metadata is properly stored."""
    dataset = get_processed_datasets(subjects=["philosophy"], split="test", subset_size=5, mode="binary")

    assert "Processed MMLU dataset" in dataset.dataset.info.description  # noqa: S101
    assert "subjects: ['philosophy']" in dataset.dataset.info.description  # noqa: S101
    assert "split: test" in dataset.dataset.info.description  # noqa: S101


def test_load_processed_dataset_from_dvc():
    """Test loading a processed dataset from dVC storage."""
    from dvc.repo import Repo

    dataset_path = "data/processed/test_binary_n100.dataset"

    repo = Repo(".")
    repo.pull(targets=[dataset_path])

    dataset = MMLUDataset.from_file(dataset_path)

    assert dataset is not None  # noqa: S101
    assert len(dataset) == 100 * 4  # noqa: S101
    assert hasattr(dataset, "dataset")  # noqa: S101
    assert "Processed MMLU dataset" in dataset.dataset.info.description  # noqa: S101
