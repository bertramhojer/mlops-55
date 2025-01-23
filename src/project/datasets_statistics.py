from collections import Counter

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from project.data import load_from_dvc


def grouped_bar_chart(data: dict[str, Counter]):
    """Create a grouped bar chart."""
    # Create bar chart with a better color scheme
    x = np.arange(4)  # the label locations
    width = 0.2  # the width of the bars
    totals = {split: sum(cnt.values()) for split, cnt in data.items()}

    colors = plt.colormaps.get_cmap("Set2")
    fig, ax = plt.subplots()
    for i, split in enumerate(data):
        cnt = data[split]
        tot = totals[split]
        ax.bar(x + i * width, [cnt[label] / tot for label in [0, 1, 2, 3]], width, label=split, color=colors(i))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Labels")
    ax.set_ylabel("Frequency (%)")
    ax.set_ylim(0, 0.5)
    ax.set_title("Labels count by split")
    ax.set_xticks(x + width / 4)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.legend()

    fig.tight_layout()
    return fig


def histogram_bar_chart(data: dict[str, Counter]):
    """Create a density plot with a line for each split."""
    fig, ax = plt.subplots()
    for split, cnt in data.items():
        y = list(cnt.values())
        sns.kdeplot(y, ax=ax, label=split)
    ax.set_xlabel("Question length")
    ax.set_ylabel("Density")
    ax.set_title("Question length distribution by split")
    ax.legend()
    fig.tight_layout()
    return fig


def main(file: str = "mmlu-balanced") -> None:
    """Compute dataset statistics."""
    _, dataset = load_from_dvc(file=file)
    dataset["train"] = datasets.Dataset.from_list(
        dataset["train"]
    )  # This line seems fine if needed for your structure

    labels = {split: dset["answer"] for split, dset in dataset.items()}
    # Rest of your code remains the same
    splits = list(dataset.keys())
    labels_count = {split: Counter(lst) for split, lst in labels.items()}

    labels_chart = grouped_bar_chart(labels_count)
    labels_chart.savefig("labels_count.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
