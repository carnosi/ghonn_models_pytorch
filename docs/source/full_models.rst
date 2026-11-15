.. _full_models:

Full Models & Components 🏗️
=============================
Complete, ready-to-train forecasting models built on top of the
:doc:`neurons` and :doc:`network_layers`, along with the supporting
components used to build them.

.. contents::
    :local:

Full Models
~~~~~~~~~~~

.. _saghonn:

SA-GHONN
^^^^^^^^
``SAGHONN`` accepts ``(batch, input_length, in_features)`` and returns
``(batch, out_features, len(predicted_feature_indices))``. Here
``out_features`` means forecast horizon, and feature 0 is predicted by default.
The default pipeline applies RevIN, moving-average decomposition, three
experts, self-attention over stacked branches, and a delta-MLP decoder.
The first-order trend expert still has a sigmoid gate and is nonlinear.
The centered decomposition and seasonal convolution use observations within
the supplied input window; the model is not a causal per-timestep encoder.

.. autoclass:: ghonn_models_pytorch.core.SAGHONN

Components
~~~~~~~~~~
Reusable building blocks used internally by SA-GHONN, and also available for
building custom architectures. The :doc:`Convolutional GHONN <network_layers>`
layers implement the temporal experts; see that page for their different
input layouts and padding behavior.

GHONU bank
^^^^^^^^^^
.. autoclass:: ghonn_models_pytorch.core.GhonuBank

HONU bank
^^^^^^^^^
.. autoclass:: ghonn_models_pytorch.core.HonuBank

RevIN
^^^^^
.. autoclass:: ghonn_models_pytorch.core.RevIN
