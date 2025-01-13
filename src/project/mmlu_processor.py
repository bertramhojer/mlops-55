from typing import Any, Literal

import datasets
import torch
from transformers import BertTokenizer


class MMLUPreprocessor:  # noqa: D101
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        mode: Literal["binary", "multiclass"] = "binary",
    ):
        """
        Initialize the MMLU preprocessor.

        Args:
            tokenizer_name: Name of the BERT tokenizer to use
            max_length: Maximum sequence length for tokenization
            mode: "binary" for separate examples per choice, "multiclass" for single example with all choices
        """
        if mode not in ["binary", "multiclass"]:
            msg = f"Mode must be 'binary' or 'multiclass', got {mode}"
            raise ValueError(msg)
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def preprocess_binary(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single MMLU example into multiple binary classification examples."""
        question = example["question"]
        choices = example["choices"]
        # Handle both string ('A', 'B', etc) and integer (0, 1, etc) answers
        correct_answer = example["answer"] if isinstance(example["answer"], int) else ord(example["answer"]) - ord("A")

        processed_examples = []
        for idx, choice in enumerate(choices):
            text = f"{question} [SEP] {choice}"
            encoded = self.tokenizer(
                text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
            )

            processed_example = {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "label": float(idx == correct_answer),
            }
            processed_examples.append(processed_example)

        return processed_examples

    def preprocess_multiclass(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a single MMLU example into one multiclass classification example."""
        question = example["question"]
        choices = example["choices"]
        # Handle both string ('A', 'B', etc) and integer (0, 1, etc) answers
        correct_answer = example["answer"] if isinstance(example["answer"], int) else ord(example["answer"]) - ord("A")

        # Combine question with all choices
        text = f"{question} [SEP] {' [SEP] '.join(choices)}"

        encoded = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "label": correct_answer,  # Integer label (0,1,2,3) for multiclass
        }

    def preprocess_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Preprocess entire MMLU dataset."""
        if self.mode == "binary":

            def process_binary(example):
                processed = self.preprocess_binary(example)
                return {
                    "input_ids": torch.stack([ex["input_ids"] for ex in processed]),
                    "attention_mask": torch.stack([ex["attention_mask"] for ex in processed]),
                    "labels": torch.tensor([ex["label"] for ex in processed]),
                }

            # Process and convert to the right format
            processed: datasets.Dataset = dataset.map(
                process_binary, remove_columns=dataset.column_names, batched=False
            )

            # Cleaner approach:
            input_ids = [idx for example in processed for idx in example["input_ids"]]
            attention_mask = [mask for example in processed for mask in example["attention_mask"]]
            labels = [label for example in processed for label in example["labels"]]

            return datasets.Dataset.from_dict(
                {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            )

        # multiclass
        def process_multiclass(example):
            processed = self.preprocess_multiclass(example)
            return {
                "input_ids": processed["input_ids"].tolist(),
                "attention_mask": processed["attention_mask"].tolist(),
                "labels": processed["label"],  # Already an integer
            }

        # Process the dataset
        processed = dataset.map(process_multiclass, remove_columns=dataset.column_names, batched=False)

        return datasets.Dataset.from_dict(
            {
                "input_ids": [example["input_ids"] for example in processed],
                "attention_mask": [example["attention_mask"] for example in processed],
                "labels": [example["labels"] for example in processed],
            }
        )

    def create_training_batch(
        self, examples: list[dict[str, torch.Tensor]], device: str = "cuda"
    ) -> dict[str, torch.Tensor]:
        """Create a batch of examples for training."""
        batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]).to(device),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]).to(device),
        }

        if self.mode == "binary":
            batch["labels"] = torch.tensor([ex["label"] for ex in examples]).float().to(device)
        else:
            batch["labels"] = torch.tensor([ex["label"] for ex in examples]).long().to(device)

        return batch
