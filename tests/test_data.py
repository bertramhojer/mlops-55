import datasets
import pytest
import torch
from transformers import AutoTokenizer

from project.data import (
    preprocess_binary,
    preprocess_dataset,
    subset_dataset,
)


# Mock settings
class MockSettings:
    """Mock settings for testing."""

    GCP_JOB = 0
    GCP_BUCKET = "mock_bucket"


# Mock dataset
@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "question": ["What is the capital of France?", "What is 2+2?"],
            "choices": [["Paris", "London", "Berlin", "Madrid"], ["3", "4", "5", "6"]],
            "answer": [0, 1],
        }
    )


# Mock tokenizer
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_subset_dataset(mock_dataset):
    """Test subsetting a dataset."""
    subset = subset_dataset(mock_dataset, subset_size=1, random_seed=42)
    if len(subset) != 1:
        msg = f"Expected subset length to be 1, but got {len(subset)}"
        raise AssertionError(msg)


def test_preprocess_binary(mock_dataset, mock_tokenizer):
    """Test preprocessing a single example."""
    example = mock_dataset[0]
    processed = preprocess_binary(example, mock_tokenizer, max_length=128)
    if len(processed) != 4:
        msg = f"Expected 4 processed examples, but got {len(processed)}"
        raise AssertionError(msg)
    if not all(isinstance(ex["input_ids"], torch.Tensor) for ex in processed):
        msg = "Not all processed examples have 'input_ids' as torch.Tensor"
        raise AssertionError(msg)


def test_preprocess_dataset(mock_dataset, mock_tokenizer):
    """Test preprocessing an entire dataset."""
    processed = preprocess_dataset(mock_dataset, mock_tokenizer, max_length=128)
    # The number of processed examples should be 2 (original examples) * 3 (binary choices per example)
    if len(processed) != len(mock_dataset) * 3:
        msg = f"Expected {len(mock_dataset) * 3} processed examples, but got {len(processed)}"
        raise AssertionError(msg)
    if "input_ids" not in processed.column_names:
        msg = "'input_ids' not found in processed column names"
        raise AssertionError(msg)
