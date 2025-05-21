"""Defines the Higher-Order Neural Units (HONU) model."""

from __future__ import annotations

import math
from itertools import combinations_with_replacement
from typing import Any

import torch
from torch import Tensor, nn

__version__ = "0.0.1"


class HONU(nn.Module):
    """Higher-Order Neural Units (HONU) model for polynomial regression.

    This model computes polynomial feature combinations of the input data and
    applies trainable weights to produce the output. It supports configurable
    polynomial orders and optional bias terms.

    Methods:
        __init__: Initializes the HONU model with the specified parameters.
        __repr__: Returns a string representation of the HONU model.
        forward: Performs a forward pass through the HONU model.
        _validate_setup: Validates the configuration of the model.
        _initialize_weights: Initializes the trainable weights of the model.
        _get_combinations: Precomputes index combinations for polynomial features.
        _get_colx: Computes the polynomial feature map for the input batch.
    """

    _comb_idx: Tensor

    def __init__(
        self,
        in_features: int,
        polynomial_order: int,
        *,
        activation: str = "identity",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the Higher-Order Neural Units model.

        Args:
            in_features (int): Number of input features.
            polynomial_order (int): Order of the HONU model.
            activation (str, optional): Activation function to be used, by default "identity".
            **kwargs: Additional keyword arguments:

                - weight_divisor (int or float, optional): Divisor for the randomly initialized
                  weights, by default 100.0.
                - bias (bool, optional): Whether to include a bias term, by default True.

        Attributes:
            order (int): Polynomial order of the model.
            in_features (int): Number of input features.
            _weight_divisor (float): Divisor used to scale the randomly initialized weights.
            _bias (bool): Indicates whether a bias term is included in the model.
            weight (nn.Parameter): Trainable weights of the model.
            _num_combinations (int): Number of polynomial feature combinations.
            _comb_idx (Tensor): Precomputed index combinations for polynomial features.
        """
        super().__init__()
        # Main model parameters
        self.order = polynomial_order
        self.in_features = in_features

        # Optional params
        weight_divisor = kwargs.get("weight_divisor", 100.0)
        if not isinstance(weight_divisor, (int, float, str)):
            msg = f"weight_divisor must be a number or string, got {type(weight_divisor)}"
            raise TypeError(msg)
        self._weight_divisor = float(weight_divisor)
        self._bias = kwargs.get("bias", True)
        self._activation = activation
        if self._activation in ["identity", "linear"]:
            self._activation_function = lambda x: x
        else:
            self._activation_function = getattr(torch.nn.functional, self._activation)
        self._validate_setup()

        # Initialize weights as trainable parameters
        self.weight = nn.Parameter(self._initialize_weights())

        # Get all combinations of indices for the polynomial features
        self._num_combinations = self.weight.size(0)
        self.register_buffer("_comb_idx", self._get_combinations())

    def __repr__(self) -> str:
        """Return a string representation of the HONU model."""
        return (
            f"HONU(in_features={self.in_features}, polynomial_order={self.order}, "
            f"bias={self._bias}, activation={self._activation})"
        )

    def _validate_setup(self) -> None:
        """Validates the configuration of the model to ensure all parameters are correctly set.

        This method checks the following conditions:
            - The `polynomial_order` must be greater than 0.
            - The `in_features` must be greater than 0.
            - The `weight_divisor` must be greater than 0.

        Raises:
            ValueError: If any of the above conditions are not met.
        """
        if self.order <= 0:
            msg = f"Polynomial order must be greater than 0. Got {self.order}."
            raise ValueError(msg)
        if self.in_features <= 0:
            msg = f"Input length must be greater than 0. Got {self.in_features}."
            raise ValueError(msg)
        if self._weight_divisor <= 0:
            msg = f"Weight divisor must be greater than 0. Got {self._weight_divisor}."
            raise ValueError(msg)

    def _initialize_weights(self) -> Tensor:
        """Initialize weights for the HONU model.

        This method initializes the weights for the model based on the input length,
        polynomial order, and whether a bias term is included. The number of weights
        is calculated using the formula for combinations with repetition:
            Combinations with repetition = ((n + r - 1)! / (r! * (n - 1)!))
        where:
            - n is the number of states, calculated as the input length + 1 if a bias is included.
            - r is the polynomial order of the neuron.

        Returns:
            Array of initialized weights.
        """
        # Calculate the number of weights needed based on the order and input length
        n_weights = self.in_features + 1 if self._bias else self.in_features
        num_weights = int(
            math.factorial(n_weights + self.order - 1)
            / (math.factorial(self.order) * math.factorial(n_weights - 1))
        )
        # Initialize weights randomly and scale them
        return torch.rand(num_weights) / self._weight_divisor

    def _get_combinations(self) -> Tensor:
        """Precompute and return all index combinations for the given input length and order.

        This method generates combinations with replacement of indices based on the input
        length and polynomial order. If bias is included, an additional feature is accounted
        for in the combinations. The resulting combinations are stored as a tensor.

        Returns:
            Tensor: A tensor containing all index combinations with shape
            (num_combinations, order).
        """
        # Precompute all index combinations once and store as buffer
        n_feat = self.in_features + (1 if self._bias else 0)
        return torch.tensor(
            list(combinations_with_replacement(range(n_feat), self.order)),
            dtype=torch.long,
        )  # shape: (num_combinations, order)

    def _get_colx(self, x: Tensor) -> Tensor:
        """Compute polynomial feature map using precomputed index combinations.

        For each sample in the batch, generates all degree-`order` monomials
        (with replacement) of the input features. If `bias=True`, a constant
        term is prepended before forming combinations.

        Args:
            x : Input batch of shape (batch_size, input_length).

        Returns:
            Tensor[B, num_combinations]: Tensor of shape (batch_size, num_combinations)
                                        where each column is the product of one
                                        combination of input features.
        """
        # Add bias column if needed
        if self._bias:
            ones = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
            x = torch.cat([ones, x], dim=1)  # now x.shape = [B, n_feat]

        # x_exp expected shape [B, num_combinations, n_feat]
        x_exp = x.unsqueeze(1).expand(-1, self._comb_idx.size(0), -1)

        # idx expected shape   [B, num_combinations, order]
        idx = self._comb_idx.unsqueeze(0).expand(x.size(0), -1, -1)

        # selected expected shape [B, num_combinations, order]
        selected = x_exp.gather(2, idx)

        # colx expected shape [B, num_combinations]
        return selected.prod(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the HONU model.

        Args:
            x: Input tensor [B, input_length] with the data.

        Returns:
            Tensor[B, 1]: Output tensor from the model.
        """
        # Get the polynomial feature map
        colx = self._get_colx(x)

        # Compute the output by multiplying the feature map with the weights
        return self._activation_function(torch.matmul(colx, self.weight.view(-1, 1)))


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
