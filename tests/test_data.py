import datasets
import pytest
import torch
from transformers import AutoTokenizer

from project.data import create_dataset_dict, dataset_to_dvc, load_from_dvc, preprocess_binary, subset_dataset


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_subset_dataset():
    """Test that subsetting the dataset returns the correct size."""
    aux_train = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
    aux_subset = subset_dataset(aux_train, subset_size=10, random_seed=42)
    assert len(aux_train) > len(aux_subset)  # noqa: S101
    assert len(aux_subset) == 10  # noqa: S101


def test_preprocess_binary(tokenizer):
    """Test that preprocess_binary returns the correct format."""
    aux_train = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
    aux_subset = subset_dataset(aux_train, subset_size=10, random_seed=42)
    preproccessed = preprocess_binary(aux_subset, tokenizer, max_length=128)

    item = preproccessed[0]
    assert isinstance(item["input_ids"], torch.Tensor)  # noqa: S101
    assert isinstance(item["attention_mask"], torch.Tensor)  # noqa: S101
    assert isinstance(item["label"], torch.Tensor)  # noqa: S101


def test_create_dataset_dict(tokenizer):
    """Test that create_dataset_dict returns the correct format."""
    aux_train = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
    aux_subset = subset_dataset(aux_train, subset_size=10, random_seed=42)
    dataset_dict = create_dataset_dict(aux_subset, aux_subset, aux_subset, tokenizer, max_length=128)

    assert isinstance(dataset_dict, datasets.DatasetDict)  # noqa: S101
    assert isinstance(dataset_dict["train"], datasets.Dataset)  # noqa: S101
    assert isinstance(dataset_dict["validation"], datasets.Dataset)  # noqa: S101
    assert isinstance(dataset_dict["test"], datasets.Dataset)  # noqa: S101


def test_dataset_to_dvc(tokenizer):
    """Test that dataset_to_dvc saves the dataset to the correct path."""
    aux_train = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
    aux_subset = subset_dataset(aux_train, subset_size=10, random_seed=42)
    dataset_dict = create_dataset_dict(aux_subset, aux_subset, aux_subset, tokenizer, max_length=128)
    dataset_to_dvc(dataset_dict, "data/processed/mmlu_test.json", remote="remote_storage")


def test_load_from_dvc():
    """Test that load_from_dvc loads the dataset from the correct path."""
    dataset_dict = load_from_dvc("data/processed/mmlu_test.json", remote="remote_storage")
    assert isinstance(dataset_dict, datasets.DatasetDict)  # noqa: S101
    assert len(dataset_dict["train"]) == 10 * 4  # noqa: S101
    assert len(dataset_dict["validation"]) == 10 * 4  # noqa: S101
    assert len(dataset_dict["test"]) == 10 * 4  # noqa: S101
