"""Defines the GHONU (Gated Higher-Order Neural Unit) model."""

from torch import Tensor, nn

from .honu import HONU

__version__ = "0.0.1"


class GHONU(nn.Module):
    """GHONU (Gated Higher-Order Neural Unit) model.

    This model combines two Higher-Order Neural Units (HONUs): a predictor HONU and a gate HONU.
    The gate HONU modulates the output of the predictor HONU using a specified activation function.

    Methods:
        __repr__(): Returns a string representation of the GHONU model.
        forward(x: Tensor, *, return_elements=False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
            Performs the forward pass of the GHONU model.
    """
    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        predictor_order: int,
        gate_order: int,
        *,
        gate_activation: str = "sigmoid",
        weight_divisor: int = 100,
        bias: bool = True,
    ) -> None:
        """Initialize the GHONU (Gated Higher-Order Neural Unit) model.

        Args:
            in_features (int): The number of input features for the model.
            predictor_order (int): The order of the predictor HONU.
            gate_order (int): The order of the gate HONU.
            gate_activation (str, optional): The activation function to use for the gate.
                Defaults to "sigmoid". Must be a valid activation function in `torch.nn.functional`.
            weight_divisor (int, optional): A divisor applied to the weights for scaling.
                Defaults to 100.
            bias (bool, optional): Whether to include a bias term in the HONUs.
                Defaults to True.

        Attributes:
            in_features (int): The number of input features for the model.
            predictor_order (int): The order of the predictor HONU.
            gate_order (int): The order of the gate HONU.
            _weight_divisor (int): The divisor applied to the weights for scaling.
            _bias (bool): Indicates whether a bias term is included in the HONUs.
            _gate_activation (str): The activation function used for the gate.
            predictor (HONU): The predictor HONU instance.
            activation_function (Callable): The activation function for the gate.
            gate (HONU): The gate HONU instance.
        """
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

    def forward(
        self, x: Tensor, *, return_elements=False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Perform the forward pass of the GHONU model.

        Args:
            x (Tensor): Input tensor.
            return_elements (bool, optional): If True, return the individual
                outputs of the predictor and gate along with the final output.
                Defaults to False.

        Returns:
            Tensor: The final output of the model. If `return_elements` is True,
            returns a tuple containing the final output, predictor output, and
            gate output.
        """
        # Get the outputs of the predictor and gate HONUs
        predictor_output = self.predictor(x)
        gate_output = self.gate(x)

        # Apply the gate to the predictor output
        output = predictor_output * self.activation_function(gate_output)
        if return_elements:
            return output, predictor_output, gate_output

        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
