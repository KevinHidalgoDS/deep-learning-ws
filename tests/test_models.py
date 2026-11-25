"""Pruebas del Sprint 3: arquitecturas, entrenamiento sin fuga de información y optimización."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from src.config import read_config_mapping
from src.data.preprocess import hourly_grid
from src.features.build_features import run as build_features
from src.models import optimize, train_model
from src.models.architectures import (
    TCN,
    HuberLoss,
    LSTMAttention,
    QuantileLoss,
    build_model,
    point_forecast,
    seasonal_naive,
    validate_quantiles,
)
from src.models.train_model import (
    Split,
    TrainSettings,
    check_chronology,
    fit_network,
    load_network,
    predict,
)

TZ = "America/Bogota"
TARGET = "precio_bolsa"
LOOKBACK, HORIZON, N_FEATURES = 12, 4, 5
SMALL_PARAMS = {
    "lstm_attention": {"hidden_units": 8, "num_layers": 1, "dropout": 0.0},
    "tcn": {"hidden_units": 8, "levels": 2, "kernel_size": 3, "dropout": 0.0},
    "nbeats": {"hidden_units": 16, "num_blocks": 2, "layers_per_block": 2, "dropout": 0.0},
}


@pytest.fixture
def config() -> dict:
    """Configuración real con ventanas, redes y búsquedas pequeñas para pruebas rápidas."""
    cfg = copy.deepcopy(read_config_mapping())
    cfg["training"].update(
        lookback_steps=LOOKBACK, forecast_horizon=HORIZON, epochs=3, batch_size=32, device="cpu"
    )
    cfg["features"].update(lags=[1, 24], rolling_windows=[24])
    cfg["models"].update(SMALL_PARAMS)
    cfg["optimization"].update(
        n_trials=2,
        epochs=2,
        search_space={
            "learning_rate": [0.001, 0.01],
            "lookback_steps": [6, LOOKBACK, 500],  # 500 no cabe y debe descartarse.
            "hidden_units": [4, 8],
            "dropout": [0.0, 0.2],
        },
    )
    return cfg


@pytest.fixture
def project(tmp_path, config: dict):
    """Proyecto temporal con los tensores del Sprint 2 generados a partir de datos sintéticos."""
    index = hourly_grid("2026-06-01", "2026-07-10", TZ)
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
    return tmp_path


def random_split(n: int, seed: int = 0, offset_hours: int = 0) -> Split:
    """Partición sintética donde ``y`` depende linealmente del último paso de ``x``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, LOOKBACK, N_FEATURES)).astype(np.float32)
    y = np.repeat(x[:, -1, :1], HORIZON, axis=1).astype(np.float32)
    ends = pd.date_range("2026-06-01", periods=n, freq="h", tz=TZ) + pd.Timedelta(
        hours=offset_hours
    )
    return Split(x, y, ends.strftime("%Y-%m-%dT%H:%M:%S%z").to_numpy(str))


def settings(**overrides) -> TrainSettings:
    """Configuración de entrenamiento pequeña y determinista."""
    values = {
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 3,
    }
    return TrainSettings(**(values | overrides), device="cpu")


# -----------------------------------------------------------------------------
# ARQUITECTURAS Y PÉRDIDAS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["lstm_attention", "tcn", "nbeats"])
@pytest.mark.parametrize("n_outputs", [1, 3])
def test_architectures_forecast_full_horizon(name: str, n_outputs: int) -> None:
    model = build_model(
        name,
        n_features=N_FEATURES,
        lookback=LOOKBACK,
        horizon=HORIZON,
        params=SMALL_PARAMS[name],
        n_outputs=n_outputs,
    )
    assert model(torch.zeros(7, LOOKBACK, N_FEATURES)).shape == (7, HORIZON, n_outputs)


def test_tcn_convolutions_are_causal() -> None:
    torch.manual_seed(0)
    model = TCN(N_FEATURES, HORIZON, hidden_units=8, levels=3).eval()
    x = torch.randn(2, LOOKBACK, N_FEATURES)
    changed = x.clone()
    changed[:, 8] += 10.0  # Cambio en el paso 8: los pasos 0..7 no pueden enterarse.
    before, after = model.encode(x), model.encode(changed)
    assert torch.allclose(before[:, :, :8], after[:, :, :8])
    assert not torch.allclose(before[:, :, 8:], after[:, :, 8:])


def test_lstm_attention_weights_are_a_distribution_over_the_window() -> None:
    model = LSTMAttention(N_FEATURES, HORIZON, hidden_units=8, num_layers=1).eval()
    model(torch.randn(3, LOOKBACK, N_FEATURES))
    weights = model.last_attention
    assert weights.shape == (3, LOOKBACK)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))


def test_quantile_loss_is_pinball_and_point_forecast_is_the_median() -> None:
    loss = QuantileLoss([0.1, 0.5, 0.9])
    target = torch.zeros(1, 1)
    prediction = torch.tensor([[[1.0, 1.0, 1.0]]])  # Sobrestimar en 1 cuesta (1 - q).
    assert loss(prediction, target).item() == pytest.approx((0.9 + 0.5 + 0.1) / 3)
    quantile_config = {"loss": "quantile", "quantiles": [0.1, 0.5, 0.9]}
    assert point_forecast(np.array([[[1.0, 2.0, 3.0]]]), quantile_config).item() == 2.0  # noqa: PLR2004
    with pytest.raises(ValueError, match=r"0\.5"):
        validate_quantiles([0.1, 0.9])


def test_huber_loss_grows_linearly_for_price_spikes() -> None:
    loss = HuberLoss(delta=1.0)
    target = torch.zeros(1, 1)
    small, spike = (
        loss(torch.full((1, 1, 1), 0.5), target),
        loss(torch.full((1, 1, 1), 10.0), target),
    )
    assert small.item() == pytest.approx(0.125)
    assert spike.item() == pytest.approx(9.5)  # MSE daría 50.


def test_seasonal_naive_repeats_the_last_day() -> None:
    history = np.arange(48, dtype=float)[None, :]
    forecast = seasonal_naive(history, horizon=26, season=24)
    assert forecast[0, :24].tolist() == list(range(24, 48))
    assert forecast[0, 24:].tolist() == [24, 25]


# -----------------------------------------------------------------------------
# ENTRENAMIENTO
# -----------------------------------------------------------------------------
def test_cropping_keeps_the_most_recent_steps_and_the_same_targets() -> None:
    split = random_split(10)
    cropped = split.cropped(5)
    assert np.array_equal(cropped.x, split.x[:, -5:])
    assert cropped.y is split.y
    with pytest.raises(ValueError, match="lookback"):
        split.cropped(LOOKBACK + 1)


def test_validation_must_come_after_training() -> None:
    train = random_split(50)
    check_chronology(train, random_split(20, offset_hours=100))
    with pytest.raises(ValueError, match="posterior"):
        check_chronology(train, random_split(20, offset_hours=10))


def test_training_learns_and_restores_the_best_weights() -> None:
    torch.manual_seed(0)
    model = build_model(
        "nbeats",
        n_features=N_FEATURES,
        lookback=LOOKBACK,
        horizon=HORIZON,
        params=SMALL_PARAMS["nbeats"],
    )
    train, validation = random_split(256), random_split(64, seed=1, offset_hours=1000)
    result = fit_network(
        model,
        train,
        validation,
        loss_fn=HuberLoss(),
        settings=settings(),
        models_config={"loss": "huber"},
    )
    val_losses = [row["val_loss"] for row in result.history]
    assert val_losses[result.best_epoch - 1] == pytest.approx(min(val_losses))
    assert result.best_val_loss < val_losses[0]
    # El modelo devuelto es el de la mejor época, no el de la última.
    prediction = torch.from_numpy(predict(result.model, validation.x, device="cpu"))
    restored = HuberLoss()(prediction, torch.from_numpy(validation.y)).item()
    assert restored == pytest.approx(result.best_val_loss, rel=1e-4)


def test_early_stopping_cuts_training_when_validation_stops_improving() -> None:
    torch.manual_seed(0)
    model = build_model(
        "tcn", n_features=N_FEATURES, lookback=LOOKBACK, horizon=HORIZON, params=SMALL_PARAMS["tcn"]
    )
    train = random_split(128)
    # Validación sin relación con la entrada: no puede mejorar de forma sostenida.
    noise = Split(
        train.x[:32],
        np.random.default_rng(5).normal(size=(32, HORIZON)).astype(np.float32),
        train.input_end[:32],
    )
    result = fit_network(
        model,
        train,
        noise,
        loss_fn=HuberLoss(),
        settings=settings(patience=2),
        models_config={"loss": "huber"},
    )
    assert result.stopped_early
    assert len(result.history) < 30  # noqa: PLR2004


def test_run_trains_every_model_without_touching_the_test_split(project, config: dict) -> None:
    (project / "data/processed/test.npz").unlink()  # Si el entrenamiento la leyera, fallaría.
    report = train_model.run(config, root=project)

    names = {m["architecture"] for m in report["models"]}
    # Con ventanas de 12 h la persistencia de 24 h no aplica y se omite.
    assert names == {"ridge", "lstm_attention", "tcn", "nbeats"}
    maes = [m["val_cop"]["mae"] for m in report["models"]]
    assert maes == sorted(maes)
    checkpoints = project / config["paths"]["models"]["checkpoints"]
    history = pd.read_csv(checkpoints / "tcn_history.csv")
    assert {"train_loss", "val_loss", "val_mae", "learning_rate"} <= set(history.columns)

    model, meta = load_network(checkpoints / "tcn_best.pt")
    data = train_model.load_datasets(config, root=project)
    prediction = predict(model, data.validation.x, device="cpu")
    metrics = train_model.validation_metrics_cop(prediction, data, config["models"])
    tcn = next(m for m in report["models"] if m["architecture"] == "tcn")
    assert meta["lookback"] == LOOKBACK
    assert metrics["mae"] == pytest.approx(tcn["val_cop"]["mae"], rel=1e-5)


def test_optimize_exports_a_loadable_champion(project, config: dict) -> None:
    report = optimize.run(config, architectures=["tcn"], root=project)

    trained = project / config["paths"]["models"]["trained"]
    best = json.loads((trained / "best_params.json").read_text(encoding="utf-8"))
    assert {m["architecture"] for m in best["ranking"]} == {"ridge", "tcn"}
    tcn = next(m for m in best["ranking"] if m["architecture"] == "tcn")
    assert tcn["params"]["lookback_steps"] in {6, LOOKBACK}
    assert report["champion"]["architecture"] == best["ranking"][0]["architecture"]

    forecast, meta = optimize.load_champion(trained)
    validation = train_model.load_split(project / "data/processed", "validation")
    prediction = forecast(validation.x)
    assert prediction.shape == validation.y.shape
    error = train_model.regression_metrics(
        train_model.to_cop(validation.y, train_model.load_datasets(config, root=project).scalers),
        train_model.to_cop(prediction, train_model.load_datasets(config, root=project).scalers),
    )
    assert error["mae"] == pytest.approx(meta["val_cop"]["mae"], rel=1e-5)
