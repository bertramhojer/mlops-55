import pathlib

import matplotlib.pyplot as plt
import numpy as np
import typer

from project.data import load_from_dvc
from project.settings import settings


def main(datadir: str = "data/processed/mmlu_tiny.dataset", output_dir: str = "output") -> None:
    """Compute dataset statistics."""
    dataset = load_from_dvc(filepath=datadir)
    print(f"Dataset: {dataset}")

    labels_count = {split: np.unique(dset["labels"], return_counts=True) for split, dset in dataset.items()}

    # Prepare data for plotting
    splits = list(labels_count.keys())
    labels = np.unique(np.concatenate([labels_count[split][0] for split in splits]))
    counts = {split: labels_count[split][1] for split in splits}

    # Create bar chart
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    for i, split in enumerate(splits):
        ax.bar(x + i * width, counts[split], width, label=split)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("Labels count by split")
    ax.set_xticks(x + width / len(splits))
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

    # Create the output directory if it does not exist
    output_path = pathlib.Path(settings.PROJECT_DIR / output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the plot to a folder
    output_path = f"{output_dir}/labels_count_by_split.png"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
