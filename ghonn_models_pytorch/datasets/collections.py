"""This module provides API to datasets for experimentation with the repository."""

__version__ = "0.0.1"


def air_passenger() -> str:
    """Returns the URL for the Air Passenger dataset.

    This dataset is taken from the Kats Repository in the Facebook research repo,
    see [Jiang_KATS_2022]_.
    """
    return "https://raw.githubusercontent.com/facebookresearch/Kats/refs/heads/main/kats/data/air_passengers.csv"  # pylint: disable=line-too-long


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
