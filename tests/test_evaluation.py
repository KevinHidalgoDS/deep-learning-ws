"""Pruebas del Sprint 4: métricas fuera de muestra, baselines, figuras y gradientes integrados."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from src.config import read_config_mapping
from src.data.preprocess import hourly_grid
from src.features.build_features import run as build_features
from src.models import optimize, predict_model
from src.models.predict_model import (
    bounded_mape,
    direction_labels,
    directional_metrics,
    smape,
)
from src.models.train_model import BASELINE_FILE
from src.visualization import visualize
from src.visualization.visualize import integrated_gradients
from torch import nn

TZ = "America/Bogota"
TARGET = "precio_bolsa"
LOOKBACK, HORIZON = 24, 4


@pytest.fixture
def config() -> dict:
    """Configuración real con ventanas, redes y búsquedas mínimas para pruebas rápidas."""
    cfg = copy.deepcopy(read_config_mapping())
    cfg["training"].update(
        lookback_steps=LOOKBACK, forecast_horizon=HORIZON, epochs=2, batch_size=64, device="cpu"
    )
    cfg["features"].update(lags=[1, 24], rolling_windows=[24])
    cfg["models"]["tcn"].update(hidden_units=8, levels=2)
    cfg["optimization"].update(
        n_trials=1,
        epochs=2,
        search_space={
            "learning_rate": [0.001, 0.01],
            "lookback_steps": [LOOKBACK],
            "hidden_units": [8],
            "dropout": [0.0, 0.1],
        },
    )
    cfg["evaluation"].update(integrated_gradients_steps=8, dpi=50)
    return cfg


@pytest.fixture
def project(tmp_path, config: dict):
    """Proyecto temporal con tensores del Sprint 2 y un campeón TCN exportado."""
    index = hourly_grid("2026-06-01", "2026-07-15", TZ)
    hours = np.asarray(index.hour)
    rng = np.random.default_rng(0)
    hourly = pd.DataFrame(
        {
            TARGET: 700 + 200 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 20, len(index)),
            "demanda_nacional": 8e6 + 1e6 * np.sin(2 * np.pi * hours / 24),
        },
        index=index,
    )
    interim = tmp_path / config["paths"]["data"]["interim"]
    interim.mkdir(parents=True)
    hourly.rename_axis("fecha").reset_index().to_parquet(
        interim / "simem_hourly.parquet", index=False
    )
    build_features(config, root=tmp_path)
    optimize.run(config, architectures=["tcn"], root=tmp_path)
    return tmp_path


# -----------------------------------------------------------------------------
# MÉTRICAS
# -----------------------------------------------------------------------------
def test_bounded_mape_only_changes_hours_below_the_floor() -> None:
    y_true, y_pred = np.array([0.0, 100.0]), np.array([10.0, 110.0])
    # Hora 1: 10 / max(0, 50) = 20 %. Hora 2: 10 / 100 = 10 %.
    assert bounded_mape(y_true, y_pred, floor=50) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="positivo"):
        bounded_mape(y_true, y_pred, floor=0)


def test_smape_is_symmetric_and_defined_when_both_are_zero() -> None:
    assert smape(np.array([100.0]), np.array([50.0])) == pytest.approx(
        smape(np.array([50.0]), np.array([100.0]))
    )
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_direction_uses_the_last_observed_price_for_the_first_hour() -> None:
    labels = direction_labels(np.array([[110.0, 105.0, 105.0]]), np.array([100.0]))
    assert labels.tolist() == [[1, -1, 0]]


def test_directional_accuracy_skips_flat_hours_and_penalizes_flat_forecasts() -> None:
    last = np.array([100.0])
    y_true = np.array([[110.0, 105.0, 105.0, 120.0]])  # sube, baja, igual, sube
    y_pred = np.array([[120.0, 100.0, 90.0, 90.0]])  # sube, baja, baja, igual
    accuracy, matrix = directional_metrics(y_true, y_pred, last)
    # La hora sin cambio real se excluye; el pronóstico plano de la última hora es un fallo.
    assert accuracy == pytest.approx(100 * 2 / 3)
    assert matrix.loc["sube", "sube"] == 1
    assert matrix.loc["baja", "baja"] == 1
    assert matrix.loc["sube", "igual"] == 1


# -----------------------------------------------------------------------------
# EVALUACIÓN DE EXTREMO A EXTREMO
# -----------------------------------------------------------------------------
def test_evaluation_reports_champion_and_baselines_in_cop(project, config: dict) -> None:
    summary = predict_model.run(config, root=project)
    metrics_dir = project / config["paths"]["reports"]["metrics"]
    predictions = pd.read_parquet(metrics_dir / predict_model.PREDICTIONS_FILE)

    # La ventana de 24 h permite la persistencia diaria pero no la semanal.
    assert set(summary["metrics"]) == {"champion", "naive_24h", "ridge"}
    assert summary["test"]["windows"] * HORIZON == len(predictions)
    mae = (predictions["champion"] - predictions["real"]).abs().mean()
    assert summary["metrics"]["champion"]["mae"] == pytest.approx(mae)
    table = pd.read_csv(metrics_dir / predict_model.METRICS_FILE, index_col="modelo")
    expected = 1 - table.loc["champion", "mae"] / table.loc["naive_24h", "mae"]
    assert table.loc["champion", "mejora_mae_vs_naive_24h"] == pytest.approx(expected, abs=1e-4)

    # El precio real y la persistencia salen de la serie original, ya en COP/kWh.
    hourly = pd.read_parquet(project / "data/interim/simem_hourly.parquet").set_index("fecha")[
        TARGET
    ]
    sample = predictions.sample(20, random_state=0)
    assert np.allclose(sample["real"], hourly.loc[sample["fecha_objetivo"]].to_numpy(), atol=1e-3)
    yesterday = sample["fecha_objetivo"] - pd.Timedelta(hours=24)
    assert np.allclose(sample["naive_24h"], hourly.loc[yesterday].to_numpy(), atol=1e-3)


def export_champion(project, config: dict, architecture: str) -> None:
    """Fuerza el campeón entre los modelos ya optimizados (los datos sintéticos favorecen Ridge)."""
    trained = project / config["paths"]["models"]["trained"]
    ranking = json.loads((trained / optimize.BEST_PARAMS_FILE).read_text(encoding="utf-8"))[
        "ranking"
    ]
    chosen = [m for m in ranking if m["architecture"] == architecture]
    baseline = project / config["paths"]["models"]["checkpoints"] / BASELINE_FILE
    optimize.export_champion(chosen, trained, baseline)


def test_figures_and_importances_are_exported(project, config: dict) -> None:
    export_champion(project, config, "tcn")
    predict_model.run(config, root=project)
    result = visualize.run(config, root=project)

    figures = (
        project / config["paths"]["reports"]["figures"] / config["evaluation"]["figures_subdir"]
    )
    names = {p.name for p in figures.glob("*.png")}
    assert {
        "serie_real_vs_pronostico.png",
        "residuales.png",
        "error_por_hora.png",
        "mae_por_horizonte.png",
        "importancia_variables.png",
        "importancia_temporal.png",
    } <= names
    importance = pd.read_csv(
        project / config["paths"]["reports"]["metrics"] / visualize.IMPORTANCE_FILE
    )
    assert importance["importancia"].is_monotonic_decreasing
    assert result["explanation"]["samples"] > 0


def test_ridge_champion_skips_integrated_gradients(project, config: dict) -> None:
    export_champion(project, config, "ridge")
    summary = predict_model.run(config, root=project)
    result = visualize.run(config, root=project)
    assert summary["champion"]["architecture"] == "ridge"
    assert result["explanation"] is None
    assert len(result["figures"]) == 4  # noqa: PLR2004


def test_figures_require_the_evaluation_first(tmp_path, config: dict) -> None:
    with pytest.raises(FileNotFoundError, match="predict_model"):
        visualize.run(config, root=tmp_path)


# -----------------------------------------------------------------------------
# GRADIENTES INTEGRADOS
# -----------------------------------------------------------------------------
class Linear(nn.Module):
    """Modelo lineal ``(B, L, F) → (B, H, 1)``: sus gradientes integrados son exactos."""

    def __init__(self, weights: torch.Tensor, horizon: int) -> None:
        super().__init__()
        self.weights, self.horizon = weights, horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = (x * self.weights).sum(dim=(1, 2))
        return value[:, None, None].expand(-1, self.horizon, 1)


def test_integrated_gradients_are_exact_for_a_linear_model() -> None:
    torch.manual_seed(0)
    weights = torch.randn(6, 3)
    x = np.random.default_rng(0).normal(size=(4, 6, 3)).astype(np.float32)
    attributions, error = integrated_gradients(Linear(weights, horizon=5), x, steps=4)
    assert np.allclose(attributions, x * weights.numpy(), atol=1e-5)
    assert error.max() < 1e-4  # noqa: PLR2004  # Completitud: suman f(x) - f(0).
