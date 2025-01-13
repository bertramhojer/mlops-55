import datasets
import pytest
import torch

from project.mmlu_processor import MMLUPreprocessor


@pytest.fixture
def simple_mmlu_dataset():
    """Create a simple MMLU-like dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "question": ["What is the capital of France?", "What is 2+2?"],
            "choices": [["London", "Paris", "Berlin", "Madrid"], ["3", "4", "5", "6"]],
            "answer": [1, 1],  # Using integers (0-3) instead of letters
        }
    )


def test_binary_processing(simple_mmlu_dataset):
    """
    Test binary preprocessing of MMLU dataset.
    Verifies that:
    1. Each question generates 4 examples (one per choice).
    2. Labels are correct (1.0 for correct choice, 0.0 for others).
    3. Output format is correct for BERT training.
    """  # noqa: D205
    preprocessor = MMluPreprocessor(mode="binary")
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
    preprocessor = MMluPreprocessor(mode="multiclass")
    processed = preprocessor.preprocess_dataset(simple_mmlu_dataset)

    # Should have same number of examples as original
    assert len(processed) == len(simple_mmlu_dataset)  # noqa: S101

    # Check structure
    assert all(col in processed.column_names for col in ["input_ids", "attention_mask", "labels"])  # noqa: S101

    # Verify labels are correct
    assert all(label == 1 for label in processed["labels"])  # noqa: S101


def test_batch_creation():
    """
    Test creation of training batches.
    Verifies that:
    1. Batches are created correctly for both modes.
    2. Tensors have correct shapes and types.
    """  # noqa: D205
    # Test binary mode
    binary_preprocessor = MMluPreprocessor(mode="binary")
    binary_examples = [
        {"input_ids": torch.ones(128), "attention_mask": torch.ones(128), "label": 1.0},
        {"input_ids": torch.zeros(128), "attention_mask": torch.zeros(128), "label": 0.0},
    ]
    binary_batch = binary_preprocessor.create_training_batch(binary_examples, device="cpu")
    assert binary_batch["labels"].dtype == torch.float32  # noqa: S101

    # Test multiclass mode
    multiclass_preprocessor = MMluPreprocessor(mode="multiclass")
    multiclass_examples = [
        {"input_ids": torch.ones(128), "attention_mask": torch.ones(128), "label": 1},
        {"input_ids": torch.zeros(128), "attention_mask": torch.zeros(128), "label": 2},
    ]
    multiclass_batch = multiclass_preprocessor.create_training_batch(multiclass_examples, device="cpu")
    assert multiclass_batch["labels"].dtype == torch.int64  # noqa: S101


def test_mmlu_integration():
    """Integration test with real MMLU data."""
    from project.mmlu_loader import load_mmlu_dataset

    # Load a small amount of real MMLU data
    dataset = load_mmlu_dataset(subjects=["anatomy"], split="test", subset_size=5)

    # Verify the loaded data structure
    assert "question" in dataset.column_names  # noqa: S101
    assert "choices" in dataset.column_names  # noqa: S101
    assert "answer" in dataset.column_names  # noqa: S101
    assert len(dataset) == 5  # noqa: S101

    # Test binary preprocessing
    binary_preprocessor = MMluPreprocessor(mode="binary")
    binary_processed = binary_preprocessor.preprocess_dataset(dataset)

    # Should have 4 examples per question
    assert len(binary_processed) == len(dataset) * 4  # noqa: S101
    assert "input_ids" in binary_processed.column_names  # noqa: S101
    assert "labels" in binary_processed.column_names  # noqa: S101

    # Test multiclass preprocessing
    multiclass_preprocessor = MMluPreprocessor(mode="multiclass")
    multiclass_processed = multiclass_preprocessor.preprocess_dataset(dataset)

    # Should have same number of examples as input
    assert len(multiclass_processed) == len(dataset)  # noqa: S101
    assert "input_ids" in multiclass_processed.column_names  # noqa: S101
    assert "labels" in multiclass_processed.column_names  # noqa: S101

    # Create individual example dictionaries for batch creation
    example_batches = [
        {
            "input_ids": torch.tensor(multiclass_processed["input_ids"][i]),
            "attention_mask": torch.tensor(multiclass_processed["attention_mask"][i]),
            "label": multiclass_processed["labels"][i],  # Single label for each example
        }
        for i in range(2)  # Take first two examples
    ]

    # Verify we can create a training batch
    training_batch = multiclass_preprocessor.create_training_batch(example_batches, device="cpu")

    # Verify the batch structure
    assert isinstance(training_batch["input_ids"], torch.Tensor)  # noqa: S101
    assert isinstance(training_batch["attention_mask"], torch.Tensor)  # noqa: S101
    assert isinstance(training_batch["labels"], torch.Tensor)  # noqa: S101
    assert training_batch["labels"].dtype == torch.long  # Should be long for multiclass  # noqa: S101
    assert len(training_batch["labels"]) == 2  # Should have 2 labels for our 2 examples  # noqa: S101
