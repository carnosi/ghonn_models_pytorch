"""GHONN - A Python package for Gated Higher Order Neural Networks."""

__version__ = "0.3.0"

from .core import (
    GHONN,
    GHONU,
    HONN,
    HONU,
    SAGHONN,
    ConvGhonn,
    ConvGhonn1D,
    GhonuBank,
    HonuBank,
    RevIN,
)
from .datasets import load_example_dataset

__all__ = [
    "GHONN",
    "GHONU",
    "HONN",
    "HONU",
    "SAGHONN",
    "ConvGhonn",
    "ConvGhonn1D",
    "GhonuBank",
    "HonuBank",
    "RevIN",
    "load_example_dataset",
]
