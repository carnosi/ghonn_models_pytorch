.. _insight:

Insights 💡
===========

1. GHONU and GHONN training

   The GHONU model may profit from having different initial learning rates
   for the predictor and gate weights. This can be achieved by passing the
   params to desired optimizer with various learning rates. Both GHONU and
   GHONN provide named parameters for easy access.

   Where in the case of GHONU:

   .. code-block:: python

      predictor_params = list(ghonu_neuron.predictor.parameters())
      gate_params = list(ghonu_neuron.gate.parameters())

      # Create optimizer with different learning rates for predictor and gate
      optimizer = torch.optim.Adam([
          {"params": predictor_params, "lr": 1e-3},
          {"params": gate_params, "lr": 1e-4}
      ])

   And in the case of GHONN:

   .. code-block:: python

      predictor_params = list(ghonn_layer.predictors.parameters())
      gate_params = list(ghonn_layer.gates.parameters())

      # Create optimizer with different learning rates for predictor and gate
      optimizer = torch.optim.Adam([
          {"params": predictor_params, "lr": 1e-3},
          {"params": gate_params, "lr": 1e-4}
      ])

   The difference in the API is the `s` at the end of `predictor` and `gate` in the GHONN case. This is because
   the GHONN layer may contain multiple predictors and gates, while the GHONU neuron contains only one.