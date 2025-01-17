import typer

from project.data import MMLUDataset


def main(datadir: str = "data/processed/test_binary_n100.dataset") -> None:
    """Compute dataset statistics."""
    dataset = MMLUDataset.from_file(filepath=datadir)
    print(f"Dataset: {dataset}")

    print("\n")


if __name__ == "__main__":
    typer.run(main)
