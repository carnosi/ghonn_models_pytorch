.. _quickstart:

Quick Start 🚀
================
1. Create polynomial neurons
    Polynomial neurons, such as HONU and GHONU, are implemented as PyTorch nn.Module classes. You can easily create them as follows:

    .. code-block:: python

        import ghonn_models_pytorch as gmp

        kwargs = {
            "weight_divisor": 100,  # Scales random initialization weights
            "bias": True            # Whether to use a bias term in the model
        }

        # Create a Higher Order Neural Unit (HONU) with 3 inputs and degree 2
        honu_neuron = gmp.HONU(
            in_features=3,          # Number of input features
            order=2,                # Degree of the polynomial
            activation="identity",  # Activation function
            **kwargs
        )

        # Create a Gated Higher Order Neural Unit (GHONU) with 3 inputs, predictor order 2 and gate order 3
        ghonu_neuron = gmp.GHONU(
            in_features=3,                      # Number of input features
            predictor_order=2,                  # Degree of the predictor polynomial
            gate_order=3,                       # Degree of the gate polynomial
            predictor_activation="identity",    # Predictor activation function
            gate_activation="sigmoid",          # Gate activation function
            **kwargs
        )

2. Create a network layer
    You can combine polynomial neurons into a network layer, which is also a PyTorch nn.Module. Creating a network layer is straight forward:

    .. code-block:: python

        import ghonn_models_pytorch as gmp

        kwargs = {
            "weight_divisor": 100,
            "bias": True
        }

        # Create single HONU based layer - HONN with 4 neurons of different orders and activation functions.
        honn_layer = gmp.HONN(
            in_features=3,                          # Number of input features
            out_features=2,                         # Number of output features
            layer_size=4,                           # Number of neurons in the layer
            polynomial_orders=(2, 3),               # Degree of the polynomials; cycles if shorter than layer_size
            activations=("identity", "sigmoid"),    # Activation functions for the neurons in the layer. If shorter work like a rolling buffer
            output_type="linear",                   # Output type of the layer. Can be "linear" or "sum" or "raw"
            **kwargs
        )

        # Create single GHONU based layer - GHONN with 4 neurons of different orders and activation functions.
        ghonn_layer = gmp.GHONN(
            in_features=3,                              # Number of input features
            out_features=2,                             # Number of output features
            layer_size=4,                               # Number of neurons in the layer
            predictor_orders=(2, 3),                    # Degree of the predictor polynomials in the layer. If shorter than layer size it works as rolling buffer
            gate_orders=(3, 5),                         # Degree of the gate polynomials in the layer. If shorter than layer size it works as rolling buffer
            predictor_activations="identity",           # Activation functions for the predictor neurons in the layer. If shorter work like a rolling buffer
            gate_activations=("identity", "sigmoid"),   # Activation functions for the gate neurons in the layer. If shorter work like a rolling buffer
            output_type="linear",                       # Output type of the layer. Can be "linear" or "sum" or "raw"
            **kwargs
        )

3. Create the complete SA-GHONN forecasting network
    SA-GHONN decomposes the normalized sequence into trend, seasonal, and
    residual signals. The defaults use a first-order predictor with a sigmoid gate, a
    16-kernel Conv-GHONU with 9-step kernels, and a temporal GHONU expert;
    one eight-head attention layer mixes the branch stack before a delta-MLP
    decoder.

    .. code-block:: python

        import torch
        import ghonn_models_pytorch as gmp

        # Forecast 12 steps from a 24-step, 3-feature input sequence.
        sa_ghonn = gmp.SAGHONN(
            input_length=24,
            in_features=3,
            out_features=12,
        )

        sequence = torch.randn(8, 24, 3)
        forecast = sa_ghonn(sequence)
        # forecast.shape == (8, 12, 1); feature 0 is predicted by default.

4. 🎉 Congratulations!
    You've successfully created polynomial neurons, a network layer, and a full SA-GHONN forecasting model!
    You can now integrate them into your PyTorch models or even train single polynomial units, layers, or the full model directly.
    Sometimes a single neuron or layer might be sufficient for your task, while other times the full SA-GHONN model is what you need.

    .. code-block:: python

        import torch

        # Synthetic regression data for the two-output HONN above.
        data = torch.randn(32, 3)
        targets = torch.randn(32, 2)
        batch_size = 8
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(honn_layer.parameters(), lr=1e-3)
        honn_layer.train()
        for i in range(0, data.size(0), batch_size):
            batch = data[i:i + batch_size]
            target = targets[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = honn_layer(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

.. note::
    For additional details and practical usage, please refer to the :doc:`examples` section.