"""Module for the GHONU model."""

import torch
from torch import Tensor, nn

from .honu import HONU

__version__ = "0.0.1"


class GHONU(nn.Module):
    """Model for Gated HONU."""

    def __init__(
        self,
        in_features,
        predictor_order: int,
        gate_order: int,
        *,
        gate_activation: str = "sigmoid",
        weight_divisor: int = 100,
        bias: bool = True,
    ) -> None:
        """Initialize the GHONU model."""
        super().__init__()
        # Main model parameters
        self.in_features = in_features
        self.predictor_order = predictor_order
        self.gate_order = gate_order

        # Optional params
        self._weight_divisor = weight_divisor
        self._bias = bias
        self._gate_activation = gate_activation

        # Initialize predictor and gate HONUs
        self.predictor = HONU(
            in_features,
            predictor_order,
            weight_divisor=weight_divisor,
            bias=bias,
        )
        self.activation_function = getattr(nn.functional, self._gate_activation)
        self.gate = HONU(
            in_features,
            gate_order,
            weight_divisor=weight_divisor,
            bias=bias,
        )

    def __repr__(self) -> str:
        """Return a string representation of the GHONU model."""
        act = f"{self.activation_function.__module__}.{self.activation_function.__name__}"
        myself = (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"gate_activation={act}, "
            f")\n"
            f"predictor: {self.predictor!r}\n"
            f"gate: {self.gate!r}"
        )
        return myself

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the GHONU model."""
        # Get the outputs of the predictor and gate HONUs
        predictor_output = self.predictor(x)
        gate_output = self.gate(x)

        # Apply the gate to the predictor output
        output = predictor_output * self.activation_function(gate_output)

        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
