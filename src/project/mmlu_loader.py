"""Module for loading and processing MMLU datasets."""

import datasets
import numpy as np

# Error message constants
ERROR_NO_SUBJECTS = "Must provide at least one subject"
ERROR_INVALID_SUBSET_SIZE = "subset_size must be positive"
ERROR_SUBSET_SIZE_TOO_LARGE = "subset_size ({}) is larger than dataset size ({})"
ERROR_DATASET_LOAD = "Failed to load dataset for subject '{}': {}"
ERROR_FETCH_SUBJECTS = "Failed to fetch MMLU subjects: {}"


def load_mmlu_dataset(subjects, split="test", subset_size=None, random_seed=42):
    """
    Load MMLU dataset for specified subjects.

    Args:
        subjects: List of subject names to load
        split: Dataset split to load ("train", "test", or "validation")
        subset_size: Optional size to subset the dataset to
        random_seed: Random seed for reproducibility

    Returns:
        Combined dataset for all subjects
    """
    if not subjects:
        raise ValueError(ERROR_NO_SUBJECTS)

    if subset_size is not None and subset_size <= 0:
        raise ValueError(ERROR_INVALID_SUBSET_SIZE)

    # Load and combine datasets for each subject
    all_datasets = []
    for subject in subjects:
        try:
            dataset = datasets.load_dataset("cais/mmlu", subject, split=split)
            all_datasets.append(dataset)
        except (ValueError, ImportError, FileNotFoundError) as e:
            raise RuntimeError(ERROR_DATASET_LOAD.format(subject, str(e))) from e

    # Combine all subjects into one dataset
    combined = datasets.concatenate_datasets(all_datasets)

    # Handle subsetting if requested
    if subset_size is not None:
        if subset_size > len(combined):
            raise ValueError(ERROR_SUBSET_SIZE_TOO_LARGE.format(subset_size, len(combined)))

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Randomly select indices
        indices = np.random.choice(len(combined), size=subset_size, replace=False)
        combined = combined.select(indices)

    return combined


def list_available_subjects():
    """Get list of all available MMLU subjects."""
    try:
        info = datasets.get_dataset_config_names("cais/mmlu")
        return sorted(info)
    except (ValueError, ImportError, FileNotFoundError) as e:
        raise RuntimeError(ERROR_FETCH_SUBJECTS.format(str(e))) from e


if __name__ == "__main__":
    subjects = list_available_subjects()
    print(f"Available subjects: {subjects}")
