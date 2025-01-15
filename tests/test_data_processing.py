import datasets
import pytest
import torch
from torch.utils.data import DataLoader

from project.data import MMLUDataset, get_processed_datasets
from project.mmlu_processor import MMLUPreprocessor


def test_dataset_creation():
    """Test creating dataset."""
    dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary")
    assert len(dataset) > 0  # noqa: S101

    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)  # noqa: S101
    assert isinstance(item["attention_mask"], torch.Tensor)  # noqa: S101
    assert isinstance(item["labels"], torch.Tensor)  # noqa: S101
    assert item["labels"].dtype == torch.float32  # noqa: S101


def test_dataloader_compatibility():
    """Test dataset works with DataLoader."""
    dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 2  # noqa: S101
    assert batch["labels"].shape[0] == 2  # noqa: S101


def test_save_and_load(tmp_path):
    """Test saving and loading dataset."""
    save_path = tmp_path / "test.dataset"
    original_dataset = get_processed_datasets(subjects=["anatomy"], subset_size=10, mode="binary", save_path=save_path)

    loaded_dataset = MMLUDataset.from_file(save_path)

    assert len(original_dataset) == len(loaded_dataset)  # noqa: S101
    assert all(  # noqa: S101
        torch.equal(original_dataset[0][k], loaded_dataset[0][k]) for k in ["input_ids", "attention_mask", "labels"]
    )


@pytest.fixture
def simple_mmlu_dataset():
    """Create a simple MMLU-like dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "question": ["What is the capital of France?", "What is 2+2?"],
            "choices": [["London", "Paris", "Berlin", "Madrid"], ["3", "4", "5", "6"]],
            "answer": [1, 1],
        }
    )


def test_invalid_mode():
    """Test that invalid mode raises error."""
    with pytest.raises(ValueError, match="Mode must be 'binary' or 'multiclass'"):
        MMLUPreprocessor(mode="invalid_mode")


def test_processor_modes():
    """Test processor initialization with different valid modes."""
    binary_processor = MMLUPreprocessor(mode="binary")
    assert binary_processor.mode == "binary"  # noqa: S101

    multiclass_processor = MMLUPreprocessor(mode="multiclass")
    assert multiclass_processor.mode == "multiclass"  # noqa: S101


def test_binary_processing(simple_mmlu_dataset):
    """
    Test binary preprocessing of MMLU dataset.
    Verifies that:
    1. Each question generates 4 examples (one per choice).
    2. Labels are correct (1.0 for correct choice, 0.0 for others).
    3. Output format is correct for BERT training.
    """  # noqa: D205
    preprocessor = MMLUPreprocessor(mode="binary")
    processed = preprocessor.preprocess_dataset(simple_mmlu_dataset)

    # Should have 4 examples per original question
    assert len(processed) == len(simple_mmlu_dataset) * 4  # noqa: S101

    # Check the structure of processed data
    assert all(col in processed.column_names for col in ["input_ids", "attention_mask", "labels"])  # noqa: S101

    # Verify that each question has exactly one positive label
    labels = processed["labels"]
    for i in range(0, len(labels), 4):
        question_labels = labels[i : i + 4]
        assert sum(question_labels) == 1.0  # One correct answer  # noqa: S101
        assert question_labels[1] == 1.0  # B (index 1) is correct  # noqa: S101


def test_multiclass_processing(simple_mmlu_dataset):
    """
    Test multiclass preprocessing of MMLU dataset.
    Verifies that:
    1. Each question generates one example.
    2. Labels are correct integers (0-3).
    3. Output format is correct for BERT training.
    """  # noqa: D205
    preprocessor = MMLUPreprocessor(mode="multiclass")
    processed = preprocessor.preprocess_dataset(simple_mmlu_dataset)

    # Should have same number of examples as original
    assert len(processed) == len(simple_mmlu_dataset)  # noqa: S101

    # Check structure
    assert all(col in processed.column_names for col in ["input_ids", "attention_mask", "labels"])  # noqa: S101

    # Verify labels are correct
    assert all(label == 1 for label in processed["labels"])  # noqa: S101
