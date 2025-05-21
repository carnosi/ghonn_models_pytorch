"""Dataset loading functions."""

import inspect
from types import FunctionType
from typing import Any

import pandas as pd

from . import collections

__version__ = "0.0.1"


def _get_dataset_functions() -> dict[str, FunctionType]:
    """Return a dict mapping dataset names to their URL-returning functions."""
    return {
        name: func
        for name, func in inspect.getmembers(collections, inspect.isfunction)
        if not name.startswith("_")  # skip private/internal functions
    }


def load_example_dataset(name: str, **kwargs: dict[str, Any]) -> pd.DataFrame:
    """Unified dataset loader interface from collections module.

    This function loads datasets from the collections module based on the provided name,
    which corresponds to the function names in the collections module.

    Args:
        name: Name of the dataset to load which matches function names in the collections module.
        **kwargs: Additional keyword arguments for pandas readers.

    Returns:
        The requested dataset as a DataFrame.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    dataset_funcs = _get_dataset_functions()
    if name not in dataset_funcs:
        msg = f"Unknown dataset: {name}, available datasets are: {list(dataset_funcs.keys())}"
        raise ValueError(msg)
    url = dataset_funcs[name]()
    return pd.read_csv(url, **kwargs)


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
