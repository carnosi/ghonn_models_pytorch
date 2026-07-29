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
.. autoclass:: ghonn_models_pytorch.core.SAGHONN

Components
~~~~~~~~~~
Reusable building blocks used internally by SA-GHONN, and also available for
building custom architectures. The :doc:`Convolutional GHONN <network_layers>`
layer is also used here as SA-GHONN's optional temporal embedding.

GHONU bank
^^^^^^^^^^
.. autoclass:: ghonn_models_pytorch.core.GhonuBank

HONU bank
^^^^^^^^^
.. autoclass:: ghonn_models_pytorch.core.HonuBank

RevIN
^^^^^
.. autoclass:: ghonn_models_pytorch.core.RevIN
