import pytest

from project.mmlu_loader import list_available_subjects, load_mmlu_dataset


def test_load_single_subject():
    """Test loading a single MMLU subject."""
    dataset = load_mmlu_dataset(subjects=["anatomy"], split="test")
    assert dataset is not None  # noqa: S101
    assert len(dataset) > 0  # noqa: S101
    # Check expected columns exist
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

    # Load dataset twice with same seed
    dataset1 = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=subset_size, seed=seed)

    dataset2 = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=subset_size, seed=seed)

    # Check that both datasets contain the same examples
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
