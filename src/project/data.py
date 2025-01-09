from project.mmlu_loader import load_mmlu_dataset
from project.mmlu_processor import MMluPreprocessor


def get_processed_datasets(subjects=None, split="test", subset_size=100):
    """Load and preprocess MMLU dataset in both binary and multiclass formats.

    Args:
        subjects (list): List of MMLU subjects to load.
        split (str): Dataset split ('train', 'test', or 'validation').
        subset_size (int): Number of examples to load.

    Returns:
        tuple: (binary_dataset, multiclass_dataset)
    """
    # First load the dataset
    if subjects is None:
        subjects = ["philosophy"]
    dataset = load_mmlu_dataset(subjects=subjects, split=split, subset_size=subset_size)

    # Then process it in both binary and multiclass formats
    binary_preprocessor = MMluPreprocessor(mode="binary")
    multiclass_preprocessor = MMluPreprocessor(mode="multiclass")

    binary_dataset = binary_preprocessor.preprocess_dataset(dataset)
    multiclass_dataset = multiclass_preprocessor.preprocess_dataset(dataset)

    return binary_dataset, multiclass_dataset
