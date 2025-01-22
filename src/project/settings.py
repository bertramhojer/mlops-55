from pathlib import Path

import pydantic
import pydantic_settings
import torch
import wandb


class ProjectSettings(pydantic_settings.BaseSettings):
    """Base settings for client configurations."""

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", protected_namespaces=("settings_",))

    PROJECT_DIR: Path = Path(__file__).parent.parent.parent
    WANDB_PROJECT: str = "ModernBERTQA"
    WANDB_ENTITY: str = "mlops_55"
    WANDB_API_KEY: str = pydantic.Field(..., description="Wandb API key")
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    GCP_REGION: str = "europe-north1"
    GCP_PROJECT_ID: str
    GCP_REGISTRY: str = "mlops-55"
    GCP_JOB: bool = False
    GCP_BUCKET: str = "mlops-55-bucket"


settings = ProjectSettings()  # type: ignore  # noqa: PGH003
wandb.login()
