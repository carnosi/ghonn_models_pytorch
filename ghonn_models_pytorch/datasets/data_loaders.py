"""Dataset loading functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import collections

if TYPE_CHECKING:
    from types import FunctionType

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


class ETTh1Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return normalized ETTh1 input windows and future values of one target."""

    def __init__(  # noqa: PLR0913
        self,
        frame: pd.DataFrame,
        *,
        split: str,
        input_length: int,
        prediction_length: int,
        target: str = "OT",
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
    ) -> None:
        """Create one split using normalization fitted to the training portion."""
        if split not in {"train", "validation", "test"}:
            msg = "split must be 'train', 'validation', or 'test'."
            raise ValueError(msg)

        values = frame.drop(columns="date").to_numpy(dtype=np.float32)
        self.feature_names = tuple(frame.columns.drop("date"))
        if target not in self.feature_names:
            msg = f"Unknown target {target!r}."
            raise ValueError(msg)
        self.target_index = self.feature_names.index(target)

        train_end = int(len(values) * train_ratio)
        validation_end = int(len(values) * (train_ratio + validation_ratio))
        mean = values[:train_end].mean(axis=0)
        std = values[:train_end].std(axis=0)
        self.values = (values - mean) / np.where(std == 0, 1, std)
        self.target_mean = mean[self.target_index]
        self.target_std = std[self.target_index]
        self.input_length = input_length
        self.prediction_length = prediction_length

        target_start, target_end = {
            "train": (input_length, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, len(values)),
        }[split]
        self.first_start = target_start - input_length
        self.last_start = target_end - input_length - prediction_length
        if self.last_start < self.first_start:
            msg = "Split is too short for the requested windows."
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of complete windows in this split."""
        return self.last_start - self.first_start + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input window and its future target values."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = self.first_start + index
        end = start + self.input_length
        x = torch.from_numpy(self.values[start:end].copy())
        y = torch.from_numpy(
            self.values[
                end : end + self.prediction_length,
                self.target_index : self.target_index + 1,
            ].copy()
        )
        return x, y

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized target values back to the original ETTh1 scale."""
        return values * self.target_std + self.target_mean


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
