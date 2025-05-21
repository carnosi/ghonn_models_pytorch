"""Utils for HONU based models."""

from __future__ import annotations

from itertools import cycle, islice
from warnings import warn

__version__ = "0.0.1"


def normalize_list_to_size(
    target_size: int, input_list: list | tuple, description: str = "unknown"
) -> tuple:
    """Normalizes the provided list or tuple to match the target size.

    This function ensures that the input list or tuple is normalized to match
    the specified `target_size` by either repeating, truncating, or directly
    using the input.

    Behavior:
        - If the input is empty, a `ValueError` is raised.
        - If the input length matches `target_size`, a copy of the input is returned.
        - If the input is shorter than `target_size`, it is cyclically repeated
          until the desired length is reached.
        - If the input is longer than `target_size`, it is truncated to the
          desired length, and a warning is issued.

    Args:
        target_size (int): The desired size of the output tuple.
        input_list (list | tuple): A list or tuple of elements to normalize.
        description (str, optional): A descriptive label for the type of input
            being normalized (e.g., "predictor" or "gate"). Defaults to "unknown".

    Returns:
        tuple: A tuple of elements normalized to match the target size.

    Raises:
        ValueError: If the input is empty.

    Warnings:
        - If the input is longer than `target_size`, a warning is issued,
          and the input is truncated.
    """
    n = len(input_list)
    if n == 0:
        msg = f"{description} list is empty. Cannot continue."
        raise ValueError(msg)

    if n == target_size:
        return tuple(input_list)

    if n < target_size:
        # Cycle through the original entries until we reach target_size
        return tuple(islice(cycle(input_list), target_size))

    # If too many elements, just truncate
    msg = f"Truncating {description} list ({n}) to target size {target_size}."
    warn(msg, stacklevel=2)
    return tuple(input_list[:target_size])


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
