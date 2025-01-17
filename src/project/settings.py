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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.InitSettingsSource,
        env_settings: pydantic_settings.EnvSettingsSource,
        dotenv_settings: pydantic_settings.DotEnvSettingsSource,
        file_secret_settings: pydantic_settings.SecretsSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        """See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#customise-settings-sources."""
        _init_kwargs = {k: v for k, v in init_settings.init_kwargs.items() if v is not None}
        if _init_kwargs:
            init_settings.init_kwargs = _init_kwargs
            return init_settings, env_settings, dotenv_settings, file_secret_settings
        return env_settings, dotenv_settings, file_secret_settings


settings = ProjectSettings()  # type: ignore  # noqa: PGH003
