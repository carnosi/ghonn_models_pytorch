"""Module for the HONU model."""

__version__ = "0.0.1"


class HONU:
    """HONU model class."""

    def __init__(self, input_lenght: int, order: int = 1, weight_divisor:int = 100) -> None:
        """Initialize the HONU model.

        Args:
        input_lenght: Length of the input for which the required number of weights is calculated.
        order: Order of the HONU model, by default 1 = linear
        weight_divisor: Divisor for the randomly initialized weights, by default 100
        """
        self._order = order
        self._input_lenght = input_lenght
        self._weight_divisor = weight_divisor

if __name__ == "__main__":
    from pathlib import Path
    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)