"""Module for loading and processing MMLU (Massive Multitask Language Understanding) datasets."""

import datasets
import numpy as np

# Error message constants
ERROR_INVALID_SUBSET_SIZE = "Subset size must be positive"
ERROR_SUBSET_SIZE_TOO_LARGE = "Requested subset size ({}) is larger than dataset size ({})"
ERROR_DATASET_LOAD = "Failed to load dataset: {}"


def load_mmlu_dataset(subjects=None, split="test", subset_size=None, random_seed=42, auxiliary_train=False):
    """
    Load MMLU dataset for specified subjects.

    Args:
        subjects: List of subjects to load. If None, loads complete dataset
        split: Dataset split to load ("train", "test", or "validation")
        subset_size: Optional size to subset the dataset to
        random_seed: Random seed for reproducibility
        auxiliary_train: If True, loads the auxiliary training data instead of the main dataset

    Returns:
        MMLU dataset for specified subjects and split

    Raises:
        ValueError: If subjects list is empty, subset_size is invalid, or invalid combination of parameters
        RuntimeError: If dataset loading fails
    """
    validate_inputs(subjects, subset_size, auxiliary_train, split)

    try:
        dataset = load_dataset(subjects, split, auxiliary_train)

        if subset_size is not None:
            dataset = subset_dataset(dataset, subset_size, random_seed)

        return dataset
    except Exception as e:
        raise RuntimeError(ERROR_DATASET_LOAD.format(str(e))) from e


def validate_inputs(subjects, subset_size, auxiliary_train, split):
    """
    Validate input parameters for dataset loading.

    Args:
        subjects: List of subjects to load
        subset_size: Size to subset the dataset to
        auxiliary_train: Whether auxiliary training data is requested
        split: Dataset split to load

    Raises:
        ValueError: If any input parameters are invalid
    """
    if subset_size is not None and subset_size <= 0:
        raise ValueError(ERROR_INVALID_SUBSET_SIZE)

    if subjects is not None and not subjects:
        msg = "Must provide at least one subject"
        raise ValueError(msg)

    if auxiliary_train and split != "train":
        msg = "Auxiliary training data is only available for the 'train' split"
        raise ValueError(msg)

    if auxiliary_train and subjects is not None:
        msg = "Subject selection is not supported for auxiliary training data"
        raise ValueError(msg)


def load_dataset(subjects, split, auxiliary_train):
    """
    Load the MMLU dataset based on specified parameters.

    Args:
        subjects: List of subjects to load
        split: Dataset split to load
        auxiliary_train: Whether to load auxiliary training data

    Returns:
        Loaded dataset

    Raises:
        RuntimeError: If dataset loading fails
    """
    if auxiliary_train:
        return datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")

    if subjects is None:
        return datasets.load_dataset("cais/mmlu", "all", split=split)

    subject_datasets = []
    for subject in subjects:
        subject_datasets.append(load_subject_dataset(subject, split))

    return datasets.concatenate_datasets(subject_datasets)


def load_subject_dataset(subject, split):
    """
    Load dataset for a specific subject.

    Args:
        subject: Subject to load
        split: Dataset split to load

    Returns:
        Dataset for the specified subject

    Raises:
        RuntimeError: If loading fails for the subject
    """
    try:
        return datasets.load_dataset("cais/mmlu", subject, split=split)
    except Exception as e:
        msg = f"Failed to load subject '{subject}': {str(e)}"
        raise RuntimeError(msg) from e


def subset_dataset(dataset, subset_size, random_seed):
    """
    Create a random subset of the dataset.

    Args:
        dataset: Dataset to subset
        subset_size: Size of the subset to create
        random_seed: Random seed for reproducibility

    Returns:
        Subset of the original dataset

    Raises:
        ValueError: If subset_size is larger than dataset size
    """
    if subset_size > len(dataset):
        raise ValueError(ERROR_SUBSET_SIZE_TOO_LARGE.format(subset_size, len(dataset)))

    np.random.seed(random_seed)
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return dataset.select(indices)
