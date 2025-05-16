"""Defines the HONN (Higher-Order Neural Network) model architecture."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from .honu import HONU
from .utils import normalize_list_to_size

__version__ = "0.0.1"


class HONN(nn.Module):
    """Higher-Order Neural Network (HONN) model.

    This class implements a neural network composed of multiple HONU (Higher-Order Neuron Unit)
    in a single layer. Each unit applies a polynomial transformation to the input features,
    enabling the model to capture higher-order interactions between input variables.

    The model supports flexible configurations, including the number of units, polynomial orders
    for each unit, and different output transformation types.

    Methods:
        __init__: Initializes the HONN model with the specified parameters.
        __repr__: Returns a string representation of the HONN model.
        forward: Performs a forward pass through the HONN model.
        _assign_polynomial_orders: Adjusts the polynomial orders list to match
                the number of layers.
        _get_head: Constructs and returns the output head function based on the
                specified output type.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        out_features: int,
        layer_size: int,
        polynomial_orders: list[int],
        *,
        activations: list[str] | tuple[str] | str = "identity",
        output_type: str = "linear",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the Higher-Order Neural Network model.

        Args:
            in_features (int): Number of input features for the model.
            out_features (int): Number of output features for the model.
            layer_size (int): Number of HONU layers in the model.
            polynomial_orders (list[int]): List specifying the polynomial order for each layer.
                - If the list length is less than `layer_size`, it will be cycled to match the size.
                - If the list length is greater than `layer_size`, it will be truncated.
            activations (list[str] | tuple[str], optional): List of activation fnfor each layer.
                - If the list length is less than `layer_size`, it will be cycled to match the size.
            output_type (str, optional): Type of output transformation. Defaults to "linear".
                - "sum": Sum the outputs of all layers.
                - "linear": Apply a linear transformation to the concatenated outputs.
                - "raw": Return the raw outputs of all layers without any transformation.
            **kwargs: Additional keyword arguments passed to each HONU layer.

        Attributes:
            in_features (int): Number of input features for the model.
            out_features (int): Number of output features for the model.
            layer_size (int): Number of HONU in the model layer.
            polynomial_orders (list[int]): List of polynomial orders for each layer.
            output_type (str): Type of output transformation.
            honu (nn.ModuleList): List of HONU neurons in the model.
            head (callable): Output head function or module for processing the model's output.
        """
        super().__init__()
        # Main model parameters
        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.polynomial_orders = normalize_list_to_size(
            self.layer_size, polynomial_orders, description="honus"
        )

        # Optional params
        self.output_type = output_type
        self._validate_setup()

        # Prepare activations
        # Ensure that even single str value is passed as a list
        activations = (activations,) if isinstance(activations, str) else activations
        self.activations = normalize_list_to_size(
            self.layer_size, activations, description="neuron activations"
        )
        # Extract relevant kwargs

        # Initialize HONU neurons
        self.honu = nn.ModuleList(
            [
                HONU(in_features, order, activation=activation, **kwargs)
                for order, activation in zip(self.polynomial_orders, self.activations)
            ]
        )
        # Initialize output head
        self.head = self._get_head()

    def __repr__(self) -> str:
        """Return a string representation of the HONN model."""
        # Describe head
        if self.output_type == "sum":
            head_desc = "SummedHonuOutputs"
        elif self.output_type == "linear":
            head_desc = repr(self.head)
        elif self.output_type == "raw":
            head_desc = "RawHonuOutputs"
        else:
            head_desc = "UnknownHead"
        # Describe the model
        lines = [
            f"{self.__class__.__name__}(",
            f"  in_features={self.in_features},",
            f"  out_features={self.out_features},",
            f"  layer_size={self.layer_size},",
            f"  output_type='{self.output_type}',",
            f"  head={head_desc},",
            f"  honu={self.honu}",
        ]

        # Describe the model
        return "\n".join(lines) + "\n" + "  )\n"

    def _validate_setup(self) -> None:
        """Validates the configuration of the model to ensure all parameters are correctly set.

        This method checks the following conditions:
            - The `output_type` must be one of the supported types: "sum", "linear", or "raw".
            - The `layer_size` must be greater than 0.
            - The `out_features` must be greater than 0.
            - If `output_type` is "sum", `out_features` must be exactly 1.
            - If `output_type` is "raw", `out_features` must match the value of `layer_size`.

        Raises:
            ValueError: If any of the above conditions are not met.
        """
        supported_output_types = ["sum", "linear", "raw"]
        if self.output_type not in supported_output_types:
            msg = (
                f"Invalid output type: {self.output_type}. Must be one of {supported_output_types}."
            )
            raise ValueError(msg)

        if self.layer_size <= 0:
            msg = f"Invalid layer_size: {self.layer_size}. Must be > 0."
            raise ValueError(msg)

        if self.out_features <= 0:
            msg = f"Invalid out_features: {self.out_features}. Must be > 0."
            raise ValueError(msg)

        if self.output_type == "sum" and self.out_features != 1:
            msg = f"Invalid out_features: {self.out_features}. Must be 1 when output_type is 'sum'."
            raise ValueError(msg)

        if self.output_type == "raw" and self.out_features != self.layer_size:
            msg = (
                f"Invalid out_features: {self.out_features}. Must be {self.layer_size} when "
                "output_type is 'raw'."
            )
            raise ValueError(msg)

    def _get_head(self) -> Callable:
        """Constructs and returns the output head function based on the specified output type.

        Supported `output_type` values:
            - "sum": Returns a lambda function that computes the sum of the input tensor
                    along the last dimension.
            - "linear": Returns a fully connected layer (`nn.Linear`) with `layer_size`
                    input features and `out_features` output features.
            - "raw": Returns a lambda function that outputs the input tensor unchanged.

        Returns:
            callable: A function or module that processes the output of the model.

        Raises:
            ValueError: If the specified `output_type` is not one of the supported types.
        """
        if self.output_type == "sum":
            return lambda x: x.sum(dim=-1)
        if self.output_type == "linear":
            return nn.Linear(self.layer_size, self.out_features)
        if self.output_type == "raw":
            return lambda x: x

        msg = f"Invalid output type: {self.output_type}"
        raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Perform a forward pass through the HONN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features). The shape depends on the
            specified output_type:
                - "sum": (batch_size, out_features), where outputs are summed across layers.
                - "linear": (batch_size, out_features), where a linear transformation is applied.
                - "raw": (batch_size, layer_size * out_features), where raw outputs are returned.
        """
        output = torch.stack([self.honu[i](x) for i in range(self.layer_size)], dim=-1)

        # Apply the output head
        if self.output_type == "linear":
            output = self.head(output.view(x.size(0), -1))
        elif self.output_type == "raw":
            output = output.view(x.size(0), -1)
        else:
            output = self.head(output)

        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
