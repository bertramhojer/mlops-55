import os
import typing
from copy import copy
from numbers import Number

import pydantic
import rich
import yaml
from dotenv import load_dotenv
from omegaconf import (
    DictConfig,
    ListConfig,
    OmegaConf,
    open_dict,
)
from rich.syntax import Syntax
from rich.tree import Tree

M = typing.TypeVar("M", bound=pydantic.BaseModel)


def hydra_to_pydantic(config: DictConfig, config_model: type[M]) -> M:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config_dict = typing.cast(dict[str, typing.Any], OmegaConf.to_object(config))
    return config_model(**config_dict)


def pprint_config(
    config: DictConfig,
    fields: None | list[str] = None,
    resolve: bool = True,
    exclude: None | list[str] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        exclude (List[str], optional): fields to exclude.
    """
    config = copy(config)

    style = "dim"
    tree = Tree(":gear: CONFIG", style=style, guide_style=style)
    if exclude is None:
        exclude = []

    fields_list: list[str] = fields or [str(k) for k in config]
    fields_list = list(filter(lambda x: x not in exclude, fields_list))

    with open_dict(config):
        base_config = {}
        for field in copy(fields_list):
            field_value = config.get(field)
            if field_value is None or isinstance(field_value, bool | str | Number | list | ListConfig):
                base_config[field] = copy(field_value)
                fields_list.remove(field)
        config["__root__"] = base_config
    fields_list = ["__root__"] + fields_list

    for field in fields_list:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            pyobj = OmegaConf.to_container(config_section, resolve=resolve)
            if exclude:
                _prune_keys(pyobj, exclude)
            branch_content = yaml.dump(pyobj)
        else:
            branch_content = str(config_section)

        branch.add(Syntax(branch_content, "yaml", indent_guides=True, word_wrap=True))

    rich.print(tree)


def _prune_keys(x: typing.Any | dict | list, exclude: list[str]) -> None:  # noqa: ANN401
    """Prune keys from a dict or list."""
    if isinstance(x, dict):
        for key in list(x.keys()):
            if key in exclude:
                x.pop(key)
            else:
                _prune_keys(x[key], exclude)
    elif isinstance(x, list):
        for item in x:
            _prune_keys(item, exclude)


def validate_env_variables():
    """loads and validates environment variables."""
    load_dotenv()

    # Check if os.environ has WANDB_PROJECT, WANDB_ENTITY, WANDB_API_KEY
    if not all(
        key in os.environ and os.environ[key]
        for key in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY"]
    ):
        raise ValueError("Please set WANDB_PROJECT, WANDB_ENTITY, WANDB_API_KEY in environment variables. Consider creating a .env file.")

