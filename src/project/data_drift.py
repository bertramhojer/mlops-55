from typing import TYPE_CHECKING

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import model
from omegaconf import DictConfig

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import pydantic
import pydantic_settings
import seaborn as sns
import torch
import typing
from lightning import Trainer
from project.collate import collate_fn
from project.data import load_from_dvc
from project.model import ModernBERTQA
from project.settings import settings
from project.settings import PROJECT_DIR
from project.tools import hydra_to_pydantic, pprint_config

if TYPE_CHECKING:
    import datasets


class DriftConfig(pydantic_settings.BaseSettings):
    """Configuration for running experiements."""

    batch_size: int = pydantic.Field(32, description="Batch size")
    model_name: str = pydantic.Field("answerdotai/ModernBERT-base", description="Name of model")
    n_train_samples: int | None = pydantic.Field(None, description="Number of training samples")
    n_test_samples: int | None = pydantic.Field(None, description="Number of validation samples")
    file_name: str = pydantic.Field("mmlu", description="Name of the file in DVC")

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", frozen=True, arbitrary_types_allowed=True)


@hydra.main(version_base="1.3", config_path=str(PROJECT_DIR / "configs"), config_name="drift_config")
def run(cfg: DictConfig) -> None:
    """Run drift detection loop."""
    config: DriftConfig = hydra_to_pydantic(cfg, DriftConfig)
    pprint_config(cfg)
    check_data_drift(config)


if torch.cuda.is_available() and torch.version.cuda.split(".")[0] == "11":  # type: ignore  # noqa: PGH003
    # Will enable run on certain servers, do no delete
    import torch._dynamo  # noqa: F401

    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # noqa: SLF001
    

def check_data_drift(config: DriftConfig):
    """Check data drift between train and test data."""
    dataset, _ = load_from_dvc(config.file_name)
    train_dataset: datasets.Dataset = dataset["train"]
    test_dataset: datasets.Dataset = dataset["test"]
    if config.n_train_samples:
        train_dataset = train_dataset.select(range(config.n_train_samples))
    if config.n_test_samples:
        test_dataset = test_dataset.select(range(config.n_test_samples))

    train_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, train_dataset),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, test_dataset),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if config.model_name.endswith(".ckpt"):
        transformer_model = ModernBERTQA.load_from_checkpoint(config.model_name)
    else:
        transformer_model = ModernBERTQA(
            config.model_name,
        )

    trainer = Trainer(
        accelerator=str(settings.DEVICE),
        precision="32",
        strategy="auto",
        devices=torch.cuda.device_count(),
    )
    train_attn_masks = torch.stack([torch.tensor(x) for x in train_dataset["attention_mask"]])
    test_attn_masks = torch.stack([torch.tensor(x) for x in test_dataset["attention_mask"]])

    train_preds = trainer.predict(transformer_model, train_loader)
    test_preds = trainer.predict(transformer_model, test_loader)

    train_logits = torch.cat([pred.logits for pred in train_preds], dim=0)
    test_logits = torch.cat([pred.logits for pred in test_preds], dim=0)
    train_pos_probs = torch.nn.functional.softmax(train_logits, dim=-1)[:, 1]
    test_pos_probs = torch.nn.functional.softmax(test_logits, dim=-1)[:, 1]

    train_hidden_states = torch.cat([pred.hidden_states[0] for pred in train_preds], dim=0)
    test_hidden_states = torch.cat([pred.hidden_states[0] for pred in test_preds], dim=0)

    train_hidden_states = (train_hidden_states * train_attn_masks.unsqueeze(-1)).sum(dim=1) / train_attn_masks.sum(
            dim=1, keepdim=True
    )
    test_hidden_states = (test_hidden_states * test_attn_masks.unsqueeze(-1)).sum(dim=1) / test_attn_masks.sum(
            dim=1, keepdim=True
    )

    train_df = pd.DataFrame(train_hidden_states.numpy(), columns=[f"emb_{i}" for i in range(train_hidden_states.shape[1])])
    test_df = pd.DataFrame(test_hidden_states.numpy(), columns=[f"emb_{i}" for i in range(test_hidden_states.shape[1])])

    column_mapping = ColumnMapping(
        embeddings={'emb': train_df.columns.tolist()}
    )
    
    report = Report(metrics=[
        EmbeddingsDriftMetric('emb',
                            drift_method = model(
                              threshold = 0.55,
                              bootstrap = None,
                              quantile_probability = 0.95,
                              pca_components = None,
                          )
                        )
    ])

    report.run(reference_data = train_df, current_data = test_df, 
            column_mapping = column_mapping)
    report.save_html("src/project/reports/drift_report.html")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(train_pos_probs, label="Train")
    sns.kdeplot(test_pos_probs, label="Test")
    plt.xlabel("Positive probability")
    plt.ylabel("Density")
    plt.title("Positive probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("src/project/reports/positive_probability_density.png")


if __name__ == "__main__":
    run()