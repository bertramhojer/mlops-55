import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
import hydra
import typing
from typing import TYPE_CHECKING
from omegaconf import DictConfig
import pydantic
import pydantic_settings
from loguru import logger


from project.data import load_from_dvc
from project.configs import DatasetConfig
from project.settings import settings
from project.tools import hydra_to_pydantic

if TYPE_CHECKING:
    import datasets


class VisualizeConfig(pydantic_settings.BaseSettings):
    """Configuration for running visualizations."""

    project_name: str = pydantic.Field(..., description="Name of project")
    datamodule: DatasetConfig = pydantic.Field(..., description="Dataset configuration")

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", frozen=True, arbitrary_types_allowed=True)


@hydra.main(version_base="1.3", config_path=str(settings.PROJECT_DIR / "configs"), config_name="test_config")
def visualize(cfg: DictConfig) -> None:
    """Visualize data."""
    
    project_name: str = pydantic.Field(..., description="Name of project")
    config: VisualizeConfig = hydra_to_pydantic(cfg, VisualizeConfig)

    # load data
    logger.info(f"Loading datasets from {config.datamodule.file_name}...")
    dataset, _ = load_from_dvc(config.datamodule.file_name)
    train_dataset: datasets.Dataset = dataset["train"]

    # look at data label distribution
    labels = [item["labels"] for item in train_dataset]
    label_counts = np.unique(labels, return_counts=True)
    plt.bar(label_counts[0], label_counts[1])
    plt.savefig("train_label_distribution.png")

    test_dataset: datasets.Dataset = dataset["test"]

    # look at data label distribution
    labels = [item["labels"] for item in test_dataset]
    label_counts = np.unique(labels, return_counts=True)
    plt.bar(label_counts[0], label_counts[1])
    plt.savefig("test_label_distribution.png")

    # print an example Q&A
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    for i in range(4):
        train_example = tokenizer.decode(train_dataset[i]["input_ids"]).replace("[PAD]", "")
        answer = train_dataset[i]["labels"]
        print(f"Question: {train_example}, Answer: {answer}")

if __name__ == "__main__":
    visualize()