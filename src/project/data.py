import os
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from dvc.repo import Repo
from transformers import AutoTokenizer


def subset_dataset(dataset: datasets.Dataset, subset_size: int, random_seed: int) -> datasets.Dataset:
    """Subset the dataset to a random sample of size `subset_size`."""
    if subset_size > len(dataset):
        msg = f"Subset size {subset_size} is larger than dataset size {len(dataset)}"
        raise ValueError(msg)

    np.random.seed(random_seed)
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return dataset.select(indices)


def preprocess_binary(
    example: dict[str, Any], tokenizer: AutoTokenizer, max_length: int, is_auxiliary_train: bool = False
) -> list[dict[str, torch.Tensor]]:
    """Convert a single MMLU example into multiple binary classification examples."""
    # Handle nested structure for auxiliary_train
    if is_auxiliary_train:
        example = example["train"]

    question = example["question"]
    choices = example["choices"]
    # Handle both string ('A', 'B', etc) and integer (0, 1, etc) answers
    correct_answer = example["answer"] if isinstance(example["answer"], int) else ord(example["answer"]) - ord("A")

    processed_examples = []
    for idx, choice in enumerate(choices):
        text = f"{question} [SEP] {choice}"
        encoded = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

        processed_examples.append(
            {
                "input_ids": torch.Tensor(encoded["input_ids"][0]),
                "attention_mask": torch.Tensor(encoded["attention_mask"][0]),
                "label": torch.Tensor([float(idx == correct_answer)]),
            }
        )

    return processed_examples


def preprocess_dataset(
    dataset: datasets.Dataset, tokenizer: AutoTokenizer, max_length: int, is_auxiliary_train: bool = False
) -> datasets.Dataset:
    """Preprocess entire MMLU dataset."""

    def process_binary(example: dict[str, Any]) -> dict[str, torch.Tensor]:
        processed = preprocess_binary(example, tokenizer, max_length, is_auxiliary_train)
        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in processed]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in processed]),
            "labels": torch.tensor([ex["label"] for ex in processed]),
        }

    # Process dataset
    processed = dataset.map(
        process_binary, remove_columns=dataset.column_names, batched=False, desc="Processing examples"
    )

    # First convert lists to tensors, then flatten
    flattened = {
        "input_ids": [tensor for example in processed["input_ids"] for tensor in example],
        "attention_mask": [tensor for example in processed["attention_mask"] for tensor in example],
        "labels": [label for example in processed["labels"] for label in example],
    }

    return datasets.Dataset.from_dict(flattened)


def create_dataset_dict(
    train: datasets.Dataset,
    validation: datasets.Dataset,
    test: datasets.Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    subset_size: int | None = None,
    random_seed: int = 42,
) -> tuple[datasets.DatasetDict, datasets.DatasetDict]:
    """Create two dataset dictionaries: one processed for training and one with original data."""
    if subset_size is not None:
        train = subset_dataset(train, subset_size=subset_size, random_seed=random_seed)

    # Create the processed dataset for training
    processed_dataset = datasets.DatasetDict(
        {
            "train": preprocess_dataset(
                dataset=train, tokenizer=tokenizer, max_length=max_length, is_auxiliary_train=True
            ),
            "validation": preprocess_dataset(
                dataset=validation, tokenizer=tokenizer, max_length=max_length, is_auxiliary_train=False
            ),
            "test": preprocess_dataset(
                dataset=test, tokenizer=tokenizer, max_length=max_length, is_auxiliary_train=False
            ),
        }
    )

    # Create the raw dataset for statistics
    raw_dataset = datasets.DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test,
        }
    )

    return processed_dataset, raw_dataset


def dataset_to_dvc(
    processed_data: datasets.DatasetDict, raw_data: datasets.DatasetDict, save_path: str, remote: str = "remote_storage"
) -> None:
    """Save both processed and raw datasets to a DVC remote."""
    save_path = Path(save_path)
    processed_path = save_path / "processed"
    raw_path = save_path / "raw"

    save_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(exist_ok=True)
    raw_path.mkdir(exist_ok=True)

    processed_data.save_to_disk(processed_path)
    raw_data.save_to_disk(raw_path)

    # init dvc repo
    repo = Repo(".")

    # add datasets to dvc
    repo.add(str(save_path))

    # push to remote
    repo.push(remote=remote)

    print(f"Saved datasets to {save_path} and pushed to {remote} remote")


def load_from_dvc(filepath: str, remote: str = "remote_storage") -> tuple[datasets.DatasetDict, datasets.DatasetDict]:
    """Load both processed and raw datasets from a DVC remote."""
    if not os.path.exists(filepath):
        # init dvc repo
        repo = Repo(".")
        # pull dataset from dvc
        repo.pull(remote=remote, targets=[filepath])
    else:
        print(f"Dataset already exists at {filepath}")

    processed_path = Path("data") / "processed" / filepath
    raw_path = Path("data") / "raw" / filepath

    return (datasets.load_from_disk(processed_path), datasets.load_from_disk(raw_path))


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    @app.command()
    def create_dataset_local(
        subset_size: int | None = typer.Option(None, help="Size of the training subset."),
        filepath: str = typer.Option("data/processed/mmlu_tiny", help="Path to the dataset."),
    ):
        """Create a dataset."""

    @app.command()
    def create_dataset(
        subset_size: int | None = typer.Option(None, help="Size of the training subset."),
        filepath: str = typer.Option("mmlu_tiny", help="Path to the dataset."),
    ):
        """Create a dataset."""
        print("Loading dataset...")
        aux_train = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
        validation = datasets.load_dataset("cais/mmlu", "all", split="validation")
        test = datasets.load_dataset("cais/mmlu", "all", split="test")

        print("Creating dataset...")
        processed_dataset, raw_dataset = create_dataset_dict(
            train=aux_train,
            validation=validation,
            test=test,
            tokenizer=tokenizer,
            max_length=512,
            subset_size=subset_size,
        )
        dataset_to_dvc(processed_dataset, raw_dataset, filepath)

    @app.command()
    def load_dataset(path: str = typer.Option(..., help="Path to the dataset.")):
        """Load a dataset."""
        load_from_dvc(path)

    app()
