.. _examples:

Examples 🧪
============
This section contains a collection of examples that demonstrate how you can use the GHONN Models library.
Examples are in general provided in the form of Jupyter notebooks, which can be run interactively.
Install the ``examples`` extra to run the notebooks. Dataset downloads require
internet access.

Available Jupyter notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `01_air_passanger_prediction.ipynb <https://github.com/carnosi/ghonn_models_pytorch/blob/main/examples/01_air_passanger_prediction.ipynb>`_
- `02_etth1_saghonn_prediction.ipynb <https://github.com/carnosi/ghonn_models_pytorch/blob/main/examples/02_etth1_saghonn_prediction.ipynb>`_

The ETTh1 notebook uses a small subset, training-only normalization, and
validation checkpoint selection. It is a usage demonstration, not a reproduction
of full-dataset benchmark scores.

The Air Passenger notebook fits its scalers on training windows and targets
only. Test predictions use observed history for each one-step prediction;
the future forecast instead feeds predictions back into the input window.
Its test-loss curve is diagnostic, not a basis for selecting hyperparameters
or a checkpoint. The future forecast is illustrative and has no held-out
multi-step score. Rerun training before saving a checkpoint with this preprocessing.

Dataset loader API
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ghonn_models_pytorch.datasets.load_example_dataset


ETTh1 window dataset
~~~~~~~~~~~~~~~~~~~~
``ETTh1Dataset`` expects a frame with a ``date`` column and numeric measurement
columns. It uses chronological ratio-based splits (60/20/20 by default), not
fixed benchmark calendar splits. Normalization is fitted on training rows only.
Each sample is ``(input_length, n_features)`` paired with
``(prediction_length, 1)``. Validation/test inputs may include observations
from the preceding split; targets remain inside the selected split.
When forecasting ``OT``, pass ``predicted_feature_indices=(dataset.target_index,)``
to SA-GHONN, because its default target is feature 0.

.. autoclass:: ghonn_models_pytorch.datasets.data_loaders.ETTh1Dataset

Available Datasets
~~~~~~~~~~~~~~~~~~~~
.. automodule:: ghonn_models_pytorch.datasets.collections
    :members: