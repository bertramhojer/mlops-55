import pathlib

import matplotlib.pyplot as plt
import numpy as np
import typer

from project.data import load_from_dvc


def grouped_bar_chart(data: dict[str, np.ndarray], labels: np.ndarray):
    """Create a grouped bar chart."""
    # Create bar chart with a better color scheme
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    colors = plt.cm.get_cmap("tab10", len(data))
    fig, ax = plt.subplots()
    for i, split in enumerate(labels):
        ax.bar(x + i * width, data[split], width, label=split, color=colors(i))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("Labels count by split")
    ax.set_xticks(x + width / len(labels))
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    return fig


def histogram_bar_chart(data: dict[str, np.ndarray]):
    """Create a histogram bar chart."""
    fig, ax = plt.subplots()
    for split, values in data.items():
        ax.hist(values, bins=30, alpha=0.5, label=split)
    ax.set_xlabel("Query length")
    ax.set_ylabel("Counts")
    ax.set_title("Query length distribution by split")
    ax.legend()
    fig.tight_layout()
    return fig


def main(datadir: str = "data/processed/mmlu_tiny_raw", output_dir: str = "visuals") -> None:
    """Compute dataset statistics."""
    dataset = load_from_dvc(filepath=datadir)
    print(f"Dataset: {dataset}")

    labels_count = {split: np.unique(dset["labels"], return_counts=True) for split, dset in dataset.items()}
    query_length = {split: np.array([len(query) for query in dset["query"]]) for split, dset in dataset.items()}

    # Prepare data for plotting
    splits = list(labels_count.keys())

    # Labels distribution
    labels = np.unique(np.concatenate([labels_count[split][0] for split in splits]))
    counts = {split: labels_count[split][1] for split in splits}

    labels_chart = grouped_bar_chart(counts, labels)
    query_lengths_chart = histogram_bar_chart(query_length)

    # Create the output directory if it does not exist
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_chart.savefig(output_path / "labels_count.png")
    query_lengths_chart.savefig(output_path / "query_length_distribution.png")


if __name__ == "__main__":
    typer.run(main)
