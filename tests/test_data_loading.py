import pytest
import torch
from datasets import Dataset

from project.data import MMLUDataset, get_processed_datasets
from project.mmlu_loader import load_mmlu_dataset
from project.mmlu_processor import MMLUPreprocessor


@pytest.fixture
def sample_mmlu_data():
    """Create a small sample MMLU dataset for testing."""
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?", "Who wrote Romeo and Juliet?"],
            "choices": [["4", "3", "5", "6"], ["Shakespeare", "Dickens", "Hemingway", "Twain"]],
            "answer": ["A", "A"],
            "subject": ["math", "literature"],
        }
    )


@pytest.fixture
def preprocessor():
    """Create a MMLUPreprocessor instance."""
    return MMLUPreprocessor(max_length=32)


def test_mmlu_loader_validation():
    """Test input validation in load_mmlu_dataset."""
    with pytest.raises(ValueError, match="subset_size must be positive"):
        load_mmlu_dataset(subset_size=-1)

    with pytest.raises(ValueError, match="Must provide at least one subject"):
        load_mmlu_dataset(subjects=[])


def test_preprocessor_binary_mode(sample_mmlu_data, preprocessor):
    """Test binary preprocessing of MMLU data."""
    processed = preprocessor.preprocess_dataset(sample_mmlu_data)

    assert isinstance(processed, Dataset)  # noqa: S101
    assert len(processed) == len(sample_mmlu_data) * 4  # 4 choices per question # noqa: S101
    assert all(key in processed.features for key in ["input_ids", "attention_mask", "labels"])  # noqa: S101
    assert all(isinstance(label, int | float) for label in processed["labels"])  # noqa: S101


def test_preprocessor_multiclass_mode(sample_mmlu_data):
    """Test multiclass preprocessing of MMLU data."""
    preprocessor = MMLUPreprocessor(max_length=32, mode="multiclass")
    processed = preprocessor.preprocess_dataset(sample_mmlu_data)

    assert isinstance(processed, Dataset)  # noqa: S101
    assert len(processed) == len(sample_mmlu_data)  # One example per question # noqa: S101
    assert all(label in [0, 1, 2, 3] for label in processed["labels"])  # noqa: S101


def test_mmlu_dataset_creation(tmp_path):
    """Test MMLUDataset creation and loading from file."""
    # Create a small dataset
    datasets = get_processed_datasets(
        subjects=["abstract_algebra"],  # Use a real MMLU subject
        source_split="test",
        train_size=2,
        val_size=1,
        test_size=1,
        mode="binary",
        save_path=tmp_path / "test_dataset",
        random_seed=42,
    )

    # Test dataset properties
    assert all(split in datasets for split in ["train", "validation", "test"])  # noqa: S101
    assert len(datasets["train"]) == 8  # 2 questions * 4 choices in binary mode # noqa: S101
    assert len(datasets["validation"]) == 4  # 1 question * 4 choices # noqa: S101
    assert len(datasets["test"]) == 4  # 1 question * 4 choices # noqa: S101

    # Test loading from file
    loaded_dataset = MMLUDataset.from_file(tmp_path / "test_dataset", split="train", mode="binary")
    assert len(loaded_dataset) == len(datasets["train"])  # noqa: S101


def test_dataset_sizes_by_mode(tmp_path):
    """Test that dataset sizes are correct for both binary and multiclass modes."""
    # Test binary mode
    binary_datasets = get_processed_datasets(
        subjects=["abstract_algebra"],
        source_split="test",
        train_size=2,
        val_size=1,
        test_size=1,
        mode="binary",
        save_path=tmp_path / "binary_dataset",
        random_seed=42,
    )

    assert len(binary_datasets["train"]) == 8  # 2 questions * 4 choices # noqa: S101
    assert len(binary_datasets["validation"]) == 4  # 1 question * 4 choices # noqa: S101
    assert len(binary_datasets["test"]) == 4  # 1 question * 4 choices # noqa: S101

    # Test multiclass mode
    multiclass_datasets = get_processed_datasets(
        subjects=["abstract_algebra"],
        source_split="test",
        train_size=2,
        val_size=1,
        test_size=1,
        mode="multiclass",
        save_path=tmp_path / "multiclass_dataset",
        random_seed=42,
    )

    assert len(multiclass_datasets["train"]) == 2  # Original size # noqa: S101
    assert len(multiclass_datasets["validation"]) == 1  # Original size # noqa: S101
    assert len(multiclass_datasets["test"]) == 1  # Original size # noqa: S101


def test_dataset_item_format(tmp_path):
    """Test the format of individual items in the dataset."""
    datasets = get_processed_datasets(
        subjects=["abstract_algebra"],
        source_split="test",
        train_size=1,
        val_size=1,
        test_size=1,
        mode="binary",
        save_path=tmp_path / "test_dataset",
        random_seed=42,
    )

    dataset = datasets["train"]
    item = dataset[0]

    assert isinstance(item, dict)  # noqa: S101
    assert all(key in item for key in ["input_ids", "attention_mask", "labels"])  # noqa: S101
    assert isinstance(item["input_ids"], torch.Tensor)  # noqa: S101
    assert isinstance(item["attention_mask"], torch.Tensor)  # noqa: S101
    assert isinstance(item["labels"], torch.Tensor)  # noqa: S101
    assert item["labels"].dtype == torch.float32  # Binary mode uses float32 # noqa: S101


def test_multiclass_dataset_format(tmp_path):
    """Test the format of multiclass dataset items."""
    datasets = get_processed_datasets(
        subjects=["abstract_algebra"],
        source_split="test",
        train_size=1,
        val_size=1,
        test_size=1,
        mode="multiclass",
        save_path=tmp_path / "test_dataset",
        random_seed=42,
    )

    dataset = datasets["train"]
    item = dataset[0]

    assert isinstance(item["labels"], torch.Tensor)  # noqa: S101
    assert item["labels"].dtype == torch.long  # Multiclass mode uses long (int64)  # noqa: S101
    assert item["labels"].item() in [0, 1, 2, 3]  # noqa: S101


def test_reproducibility():
    """Test that random seed ensures reproducible dataset splits."""
    kwargs = {
        "subjects": ["abstract_algebra"],
        "source_split": "test",
        "train_size": 2,
        "val_size": 1,
        "test_size": 1,
        "mode": "binary",
        "random_seed": 42,
    }

    datasets1 = get_processed_datasets(**kwargs)
    datasets2 = get_processed_datasets(**kwargs)

    # Compare input_ids of first examples in each split
    for split in ["train", "validation", "test"]:
        assert torch.equal(datasets1[split][0]["input_ids"], datasets2[split][0]["input_ids"])  # noqa: S101


def test_metadata_storage_and_loading(tmp_path):
    """Test that metadata is properly stored and loaded."""
    # Test parameters
    test_params = {
        "subjects": ["abstract_algebra"],
        "source_split": "test",
        "train_size": 2,
        "val_size": 1,
        "test_size": 1,
        "mode": "binary",
        "save_path": tmp_path / "test_dataset",
        "random_seed": 42,
    }

    # Create dataset with metadata
    datasets = get_processed_datasets(**test_params)

    # Verify metadata exists in each split
    for _split_name, dataset in datasets.items():
        # Check metadata column exists
        assert "metadata" in dataset.dataset.column_names  # noqa: S101

        # Get metadata from first item
        metadata = dataset.dataset[0]["metadata"]

        # Verify all metadata fields
        assert metadata["subjects"] == str(test_params["subjects"])  # noqa: S101
        assert metadata["source_split"] == test_params["source_split"]  # noqa: S101
        assert metadata["train_size"] == test_params["train_size"]  # noqa: S101
        assert metadata["val_size"] == test_params["val_size"]  # noqa: S101
        assert metadata["test_size"] == test_params["test_size"]  # noqa: S101
        assert metadata["mode"] == test_params["mode"]  # noqa: S101
        assert metadata["random_seed"] == test_params["random_seed"]  # noqa: S101

    # Test loading from saved file
    loaded_dataset = MMLUDataset.from_file(tmp_path / "test_dataset", split="train", mode="binary")
    loaded_metadata = loaded_dataset.dataset[0]["metadata"]

    # Verify loaded metadata matches original
    assert loaded_metadata["subjects"] == str(test_params["subjects"])  # noqa: S101
    assert loaded_metadata["mode"] == test_params["mode"]  # noqa: S101
    assert loaded_metadata["random_seed"] == test_params["random_seed"]  # noqa: S101


def test_metadata_consistency(tmp_path):
    """Test that metadata is consistent across all examples in a split."""
    datasets = get_processed_datasets(
        subjects=["abstract_algebra"],
        source_split="test",
        train_size=2,
        val_size=1,
        test_size=1,
        mode="binary",
        save_path=tmp_path / "test_dataset",
        random_seed=42,
    )

    # Check each split
    for _split_name, dataset in datasets.items():
        # Get metadata from first item
        first_metadata = dataset.dataset[0]["metadata"]

        # Verify all items in split have same metadata
        for idx in range(len(dataset)):
            assert dataset.dataset[idx]["metadata"] == first_metadata  # noqa: S101
