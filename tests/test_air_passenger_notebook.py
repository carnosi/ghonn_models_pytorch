"""Exercise the notebook's preprocessing and forecast without downloads or training."""

# Ruff's assertion/package rules do not apply to standalone pytest modules.
# ruff: noqa: INP001, S101

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

plt.switch_backend("Agg")


NOTEBOOK = Path(__file__).resolve().parents[1] / "examples/01_air_passanger_prediction.ipynb"


def _run_cell(marker: str, namespace: dict[str, Any]) -> None:
    """Execute the actual notebook cell identified by its descriptive comment."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    ]
    assert len(sources) == 1
    exec(compile(sources[0], str(NOTEBOOK), "exec"), namespace)  # noqa: S102


def _prepare(values: np.ndarray) -> dict[str, Any]:
    """Run preprocessing on a monthly series with unequal lag scaling ranges."""
    namespace = {
        "data": pd.DataFrame(
            {"Passengers": values},
            index=pd.date_range("2000-01-01", periods=len(values), freq="MS"),
        ),
        "np": np,
        "MinMaxScaler": MinMaxScaler,
    }
    _run_cell("# Prepare the data for training", namespace)
    return namespace


@pytest.mark.parametrize("scaler_name", ["scaler_x", "scaler_y"])
def test_scalers_are_unaffected_by_held_out_observations(scaler_name: str) -> None:
    """Held-out outliers must not change normalization or training examples."""
    values = np.arange(1, 51, dtype=float) ** 2
    baseline = _prepare(values)
    # Ten lags leave 40 pairs: the first 32 targets end at observation 41.
    values[42:] += 100_000
    contaminated = _prepare(values)

    for attribute in ("data_min_", "data_max_", "scale_", "min_"):
        np.testing.assert_array_equal(
            getattr(baseline[scaler_name], attribute),
            getattr(contaminated[scaler_name], attribute),
        )
    np.testing.assert_array_equal(baseline["X_train"], contaminated["X_train"])
    np.testing.assert_array_equal(baseline["y_train"], contaminated["y_train"])


class _NextValuePredictor(torch.nn.Module):
    """Predict seven passengers above the most recent original-scale observation."""

    def __init__(self, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler) -> None:
        super().__init__()
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.windows = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Record the notebook's original-scale inputs and return normalized targets."""
        window = self.scaler_x.inverse_transform(inputs.detach().cpu().numpy())
        self.windows.append(window.copy())
        predictions = self.scaler_y.transform(window[:, -1:] + 7)
        return torch.as_tensor(predictions, dtype=inputs.dtype, device=inputs.device)


@pytest.fixture
def forecast(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Run the plotting/forecast cell using a deterministic one-step predictor."""
    namespace = _prepare(np.arange(1, 51, dtype=float) ** 2)
    predictor = _NextValuePredictor(namespace["scaler_x"], namespace["scaler_y"])
    namespace.update(
        torch=torch,
        pd=pd,
        plt=plt,
        mdates=mdates,
        model=predictor,
        device=torch.device("cpu"),
        X_train_tensor=torch.tensor(namespace["X_train"], dtype=torch.float32),
        X_test_tensor=torch.tensor(namespace["X_test"], dtype=torch.float32),
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        _run_cell("# Visualize model predictions", namespace)
        namespace["forecast_windows"] = np.concatenate(
            predictor.windows[-namespace["n_future"] :], axis=0
        )
        yield namespace
    finally:
        plt.close("all")


def test_forecast_starts_with_final_observed_window(forecast: dict[str, Any]) -> None:
    """The first unseen month must consume the final observation, not predict it again."""
    np.testing.assert_allclose(
        forecast["forecast_windows"][0],
        [1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500],
        atol=0.001,
        rtol=0,
    )
    assert forecast["future_times"][0] == pd.Timestamp("2004-03-01")


def test_recursive_windows_shift_left_and_append_prediction(forecast: dict[str, Any]) -> None:
    """Each next input preserves chronology and converts every lag to its own scale."""
    np.testing.assert_allclose(
        forecast["forecast_windows"][1],
        [1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500, 2507],
        atol=0.001,
        rtol=0,
    )


def test_recursive_forecast_preserves_original_scale(forecast: dict[str, Any]) -> None:
    """Target-normalized predictions must not be reused as lag-normalized features."""
    np.testing.assert_allclose(
        forecast["future_preds"],
        2500 + 7 * np.arange(1, forecast["n_future"] + 1),
        atol=0.02,
        rtol=0,
    )
    assert len(forecast["future_times"]) == len(forecast["future_preds"])
