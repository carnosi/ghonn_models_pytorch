"""Defines the GHONN (Gated Higher Order Neural Network) model architecture."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from .ghonu import GHONU
from .utils import normalize_list_to_size

__version__ = "0.0.1"


class GHONN(nn.Module):
    """GHONN (Gated Higher Order Neural Network) model.

    This class implements a neural network composed of multiple GHONUs
    (Gated Higher Order Neural Units) in a single layer. Each unit applies polynomial
    transformations to the input data and uses a gating mechanism to modulate the output.

    The model supports flexible configurations, including the number of units, polynomial orders
    for both the predictor and gate, and the activation function for the gate and different output
    types.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        out_features: int,
        layer_size: int,
        predictor_orders: list[int],
        gate_orders: list[int],
        *,
        predictor_activations: list[str] | tuple[str] | str = "identity",
        gate_activations: list[str] | tuple[str] | str = "sigmoid",
        output_type: str = "linear",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the Gater Higher Order Neural Network model.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            layer_size (int): Number of layers in the model.
            predictor_orders (list[int]): List of predictor orders for each layer.
            gate_orders (list[int]): List of gate orders for each layer.
            predictor_activations (list[str] | tuple[str], optional): List or tuple of
                activation functions for the predictor. Defaults to ("identity").
            gate_activations (list[str] | tuple[str], optional): List or tuple of
                activation functions for the gates. Defaults to ("sigmoid").
            output_type (str, optional): Type of output layer. Defaults to "linear".
            **kwargs: Additional keyword arguments passed to the GHONU layers.

        Attributes:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            layer_size (int): Number of layers in the model.
            predictor_orders (list[int]): Normalized list of predictor orders for each layer.
            gate_orders (list[int]): Normalized list of gate orders for each layer.
            gate_activations (list[str]): Normalized list of activation functions for the gates.
            output_type (str): Type of output layer.
            ghonus (nn.ModuleList): List of GHONU layers.
            head (nn.Module): Output head module.
        """
        super().__init__()
        # Main model parameters
        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.predictor_orders = normalize_list_to_size(
            self.layer_size, predictor_orders, description="predictor"
        )
        self.gate_orders = normalize_list_to_size(self.layer_size, gate_orders, description="gate")
        # Ensure that even single str value is passed as a list for activation functions
        predictor_activations = (
            (predictor_activations,)
            if isinstance(predictor_activations, str)
            else predictor_activations
        )
        self.predictor_activations = normalize_list_to_size(
            self.layer_size, predictor_activations, description="predictor activations"
        )
        gate_activations = (
            (gate_activations,) if isinstance(gate_activations, str) else gate_activations
        )
        self.gate_activations = normalize_list_to_size(
            self.layer_size, gate_activations, description="gate activations"
        )

        # Optional params
        self.output_type = output_type
        self._validate_setup()

        # Initialize the GHONUs
        self.ghonus = nn.ModuleList(
            [
                GHONU(
                    in_features,
                    p,
                    g,
                    predictor_activation=pa,
                    gate_activation=ga,
                    **kwargs,
                )
                for p, g, pa, ga in zip(
                    self.predictor_orders,
                    self.gate_orders,
                    self.predictor_activations,
                    self.gate_activations,
                )
            ]
        )
        self.head = self._get_head()

    def __repr__(self) -> str:
        """Return a string representation of the GHONN model."""
        cls = self.__class__.__name__
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
        gate_activations = tuple(dict.fromkeys(self.gate_activations))
        predictor_activations = tuple(dict.fromkeys(self.predictor_activations))
        lines = [
            f"{cls}(",
            f"  in_features={self.in_features},",
            f"  out_features={self.out_features},",
            f"  layer_size={self.layer_size},",
            f"  output_type={self.output_type},",
            f"  predictor_activation_functions={predictor_activations},",
            f"  gate_activation_functions={gate_activations},",
            f"  head={head_desc},",
            f"  ghonus={self.ghonus}",
        ]
        # Describe the model
        return "\n".join(lines) + "\n" + ")"

    @property
    def predictors(self) -> nn.ModuleList:
        """Get the ModuleList of predictor HONUs in the GHONN model.

        Returns:
            nn.ModuleList: ModuleList of predictor HONUs.
        """
        return nn.ModuleList([ghonu.predictor for ghonu in self.ghonus])

    @property
    def gates(self) -> nn.ModuleList:
        """Get the ModuleList of gate HONUs in the GHONN model.

        Returns:
            nn.ModuleList: ModuleList of gate HONUs.
        """
        return nn.ModuleList([ghonu.gate for ghonu in self.ghonus])

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

    def forward(
        self, x: Tensor, *, return_elements: bool = False
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Perform the forward pass of the GHONN model.

        Args:
            x (Tensor): Input tensor.
            return_elements (bool, optional): If True, return the individual
                predictor and gate outputs along with the network output.
                Defaults to False.

        Returns:
            Tensor: Model output if return_elements is False.
            tuple: (output, (predictor_outputs, gate_outputs)) if return_elements is True.
        """
        if return_elements:
            outs: list[Tensor] = [ghonu(x, return_elements=True) for ghonu in self.ghonus]
            # outs: Tuple of (output, predictor_output, gate_output)
            outputs, predictor_outputs, gate_outputs = (torch.stack(t, dim=-1) for t in zip(*outs))
            # Reshape outputs for 'linear' and 'raw' output types
            if self.output_type in ["linear", "raw"]:
                outputs = outputs.view(x.size(0), -1)
                predictor_outputs = predictor_outputs.view(x.size(0), -1)
                gate_outputs = gate_outputs.view(x.size(0), -1)
            final = self.head(outputs)
            return final, (predictor_outputs, gate_outputs)

        outputs = torch.stack([ghonu(x) for ghonu in self.ghonus], dim=-1)
        # Reshape outputs for 'linear' and 'raw' output types
        if self.output_type in ["linear", "raw"]:
            outputs = outputs.view(x.size(0), -1)
        return self.head(outputs)


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
