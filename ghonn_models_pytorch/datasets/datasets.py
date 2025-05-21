"""This module provides API to datasets for experimentation with the repository."""

__version__ = "0.0.1"


def get_air_passenger_dataset_url() -> str:
    """Returns the URL for the Air Passenger dataset.

    This dataset is taken from the Kats Repository in the Facebook research repo,
    see [Jiang_KATS_2022]_.
    """
    return "https://raw.githubusercontent.com/facebookresearch/Kats/refs/heads/main/kats/data/air_passengers.csv"
