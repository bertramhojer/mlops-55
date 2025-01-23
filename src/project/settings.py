from pathlib import Path

import pydantic
import pydantic_settings
import torch


class ProjectSettings(pydantic_settings.BaseSettings):
    """Base settings for client configurations."""

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", protected_namespaces=("settings_",))

    PROJECT_DIR: Path = Path(__file__).parent.parent.parent
    WANDB_PROJECT: str | None = "ModernBERTQA"
    WANDB_ENTITY: str | None = "mlops_55"
    WANDB_API_KEY: str | None = pydantic.Field(default=None, description="Wandb API key")
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    GCP_REGION: str = "europe-north1"
    GCP_PROJECT_ID: str = "flash-rock-447808-n2"
    GCP_REGISTRY: str = "mlops-55"
    GCP_JOB: bool = False
    GCP_BUCKET: str = "mlops-55-bucket"


settings = ProjectSettings()  # type: ignore  # noqa: PGH003
if all([settings.WANDB_PROJECT, settings.WANDB_ENTITY, settings.WANDB_API_KEY]):
    wandb.login()
else:
    logger.warning("Wandb not configured. Skipping login.")
