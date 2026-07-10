"""GHONN - A Python package for Gated Higher Order Neural Networks."""

__version__ = "0.2.0"

from .core import GHONN, GHONU, HONN, HONU
from .datasets import load_example_dataset

__all__ = ["GHONN", "GHONU", "HONN", "HONU", "load_example_dataset"]
