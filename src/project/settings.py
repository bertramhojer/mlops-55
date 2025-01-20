from pathlib import Path

import pydantic
import pydantic_settings
import torch


class ProjectSettings(pydantic_settings.BaseSettings):
    """Base settings for client configurations."""

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", protected_namespaces=("settings_",))

    PROJECT_DIR: Path = Path(__file__).parent.parent.parent
    WANDB_PROEJECT: str = "mlops_55"
    WANDB_ENTITY: str = "mlops_55"
    WANDB_API_KEY: str = pydantic.Field(..., description="Wandb API key")
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


settings = ProjectSettings()  # type: ignore  # noqa: PGH003
