"""Module for loading and processing MMLU datasets."""

import os

import datasets
import numpy as np

# Error message constants
ERROR_INVALID_SUBSET_SIZE = "subset_size must be positive"
ERROR_SUBSET_SIZE_TOO_LARGE = "subset_size ({}) is larger than dataset size ({})"
ERROR_DATASET_LOAD = "Failed to load MMLU dataset: {}"
ERROR_FETCH_SUBJECTS = "Failed to fetch MMLU subjects: {}"


def load_mmlu_dataset(split="test", subset_size=None, random_seed=42):
    """
    Load complete MMLU dataset.

    Args:
        split: Dataset split to load ("train", "test", or "validation") d
        subset_size: Optional size to subset the dataset to
        random_seed: Random seed for reproducibility

    Returns:
        Complete MMLU dataset for specified split
    """
    if subset_size is not None and subset_size <= 0:
        raise ValueError(ERROR_INVALID_SUBSET_SIZE)

    try:
        # Load the complete dataset using the "all" configuration
        dataset = datasets.load_dataset("cais/mmlu", "all", split=split)
    except Exception as e:
        raise RuntimeError(ERROR_DATASET_LOAD.format(str(e))) from e

    # Handle subsetting if requested
    if subset_size is not None:
        if subset_size > len(dataset):
            raise ValueError(ERROR_SUBSET_SIZE_TOO_LARGE.format(subset_size, len(dataset)))

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Randomly select indices
        indices = np.random.choice(len(dataset), size=subset_size, replace=False)
        dataset = dataset.select(indices)

    return dataset


def list_available_subjects():
    """Get list of all available MMLU subjects."""
    try:
        info = datasets.get_dataset_config_names("cais/mmlu")
        return sorted(info)
    except Exception as e:
        raise RuntimeError(ERROR_FETCH_SUBJECTS.format(str(e))) from e


if __name__ == "__main__":
    # Define the output file path
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_dataset")

    # Load the complete dataset
    dataset = load_mmlu_dataset()

    # Save the dataset to the output file
    dataset.save_to_disk(output_file)
    print(f"Dataset saved to {output_file}")
