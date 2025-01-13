import pathlib
import typing

import typer
from omegaconf import DictConfig, OmegaConf

from project.tools import pprint_config

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Create Typer apps
cli = typer.Typer(help="CLI for managing training and configuration.")
describe_app = typer.Typer(help="Describe configurations and experiments in the config directory.")
list_app = typer.Typer(help="List yaml files in the config directory.")
cli.add_typer(describe_app, name="describe")
cli.add_typer(list_app, name="list")


def print_tree(directory: pathlib.Path, prefix: str = ""):
    """Recursively print directory contents in a tree structure."""
    files = list(directory.glob("*.yaml"))
    for i, path in enumerate(files):
        connector = "└──" if i == len(files) - 1 else "├──"
        typer.echo(f"{prefix}{connector} {path.name}")

    subdirs = [d for d in directory.iterdir() if d.is_dir() and d.name not in ["experiments", "outputs"]]
    for i, subdir in enumerate(subdirs):
        connector = "└──" if i == len(subdirs) - 1 else "├──"
        typer.echo(f"{prefix}{connector} {subdir.name}/")
        print_tree(subdir, prefix + ("    " if i == len(subdirs) - 1 else "│   "))


@list_app.command("experiments")
def list_experiments():
    """List all available experiment configurations."""
    experiments_path = CONFIG_DIR / "experiments"
    print_tree(experiments_path)


@list_app.command("configs")
def list_configs():
    """List all available configuration files in a tree structure."""
    print_tree(CONFIG_DIR)


@describe_app.command("config")
def describe_config(
    path: str = typer.Argument(..., help="Path to a configuration file relative to configs/"),
):
    """Show the content of a specific configuration file."""
    config_file = CONFIG_DIR / path
    if not config_file.exists() and config_file.with_suffix(".yaml").exists():
        typer.echo(f"Configuration file {path} not found!", err=True)
        raise typer.Exit(code=1)
    config = typing.cast(DictConfig, OmegaConf.load(config_file))
    pprint_config(config)


@describe_app.command("exp")
def describe_experiment(
    name: str = typer.Argument(..., help="Name of experiment in configs/experiments/"),
):
    """Describe an experiment."""
    experiment_file = CONFIG_DIR / "experiments" / name
    if not experiment_file.exists() and experiment_file.with_suffix(".yaml").exists():
        typer.echo(f"Experiment file {name} not found!", err=True)
        raise typer.Exit(code=1)
    config = typing.cast(DictConfig, OmegaConf.load(experiment_file))
    pprint_config(config)


if __name__ == "__main__":
    cli()
