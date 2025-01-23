from pathlib import Path

import pytest
import pytest_check as check

import hydra
import hydra.errors
from project.tools import hydra_to_pydantic
from project.train import ExperimentConfig

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("experiment", "should_pass"), [(None, True), ("debug", True), ("exp1", True), ("invalid_experiment", False)]
)
def test_experiment_configuration(experiment: str, should_pass: bool):
    """Test configuration loading with pass/fail scenarios."""
    with hydra.initialize_config_dir(config_dir=(PROJECT_DIR / "configs").as_posix()):
        if should_pass:
            try:
                cfg = hydra.compose(
                    config_name="train_config", overrides=[f"experiment={experiment}"] if experiment else []
                )
                converted_config: ExperimentConfig = hydra_to_pydantic(cfg, ExperimentConfig)
                check.is_instance(
                    converted_config, ExperimentConfig, "Configuration should be of type ExperimentConfig"
                )
            except (hydra.errors.ConfigCompositionException, ValueError) as e:
                pytest.fail(f"Unexpected error for experiment {experiment}: {e}")
        else:
            with pytest.raises((hydra.errors.ConfigCompositionException, ValueError)):
                hydra.compose(config_name="train_config", overrides=[f"experiment={experiment}"])
