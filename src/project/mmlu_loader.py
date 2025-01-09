"""Module for loading and processing MMLU datasets."""

import os

import datasets
import numpy as np

# Error message constants
ERROR_INVALID_SUBSET_SIZE = "subset_size must be positive"
ERROR_SUBSET_SIZE_TOO_LARGE = "subset_size ({}) is larger than dataset size ({})"
ERROR_DATASET_LOAD = "Failed to load MMLU dataset: {}"
ERROR_FETCH_SUBJECTS = "Failed to fetch MMLU subjects: {}"


def load_mmlu_dataset(subjects=None, split="test", subset_size=None, random_seed=42):
    """
    Load MMLU dataset for specified subjects.

    Args:
        subjects: List of subjects to load. If None, loads complete dataset
        split: Dataset split to load ("train", "test", or "validation")
        subset_size: Optional size to subset the dataset to
        random_seed: Random seed for reproducibility

    Returns:
        MMLU dataset for specified subjects and split

    Raises:
        ValueError: If subjects list is empty or subset_size is invalid
        RuntimeError: If dataset loading fails
    """
    if subset_size is not None and subset_size <= 0:
        raise ValueError(ERROR_INVALID_SUBSET_SIZE)

    if subjects is not None and not subjects:
        msg = "Must provide at least one subject"
        raise ValueError(msg)

    try:
        # Load either complete dataset or specific subjects
        if subjects is None:
            dataset = datasets.load_dataset("cais/mmlu", "all", split=split)
        else:
            # Load and concatenate individual subject datasets
            subject_datasets = []
            for subject in subjects:
                try:
                    subject_dataset = datasets.load_dataset("cais/mmlu", subject, split=split)
                    subject_datasets.append(subject_dataset)
                except Exception as e:
                    msg = f"Failed to load subject '{subject}': {str(e)}"
                    raise RuntimeError(msg) from e
            dataset = datasets.concatenate_datasets(subject_datasets)

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
