"""Pruebas de ausencia de fuga de información en el preprocesamiento y las características."""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from src.config import read_config_mapping
from src.data.make_dataset import ingest
from src.data.preprocess import (
    align_daily_causal,
    build_tables,
    causal_impute,
    hourly_grid,
    load_sources,
)
from src.features.build_features import (
    add_cyclical_features,
    add_target_features,
    chronological_split,
    make_windows,
    prepare_datasets,
    run,
)

TZ = "America/Bogota"
TARGET = "precio_bolsa"
LOOKBACK, HORIZON = 12, 4


@pytest.fixture
def config() -> dict:
    """Configuración real con ventanas cortas para que las pruebas sean rápidas."""
    cfg = copy.deepcopy(read_config_mapping())
    cfg["training"].update(lookback_steps=LOOKBACK, forecast_horizon=HORIZON)
    cfg["features"].update(lags=[1, 24], rolling_windows=[24])
    return cfg


def hourly_frame(days: int = 20, seed: int = 42) -> pd.DataFrame:
    """Tabla horaria sintética con estacionalidad diaria, como la de ``data/interim/``."""
    last_day = pd.date_range("2026-06-01", periods=days, freq="D")[-1]
    index = hourly_grid("2026-06-01", last_day.date().isoformat(), TZ)
    rng = np.random.default_rng(seed)
    hours = np.asarray(index.hour)
    price = 700 + 200 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 20, len(index))
    return pd.DataFrame(
        {
            TARGET: price,
            "demanda_nacional": 8e6 + 1e6 * np.sin(2 * np.pi * hours / 24),
            "aportes_hidricos": 2e8,
            "generacion_real_hidraulica": 1.6e8,
            "generacion_real_termica": 0.4e8,
        },
        index=index,
    )


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------------------------------
def test_causal_imputation_only_carries_past_values_within_limit() -> None:
    index = pd.date_range("2026-06-01", periods=8, freq="h", tz=TZ)
    frame = pd.DataFrame({"x": [np.nan, 1, np.nan, np.nan, np.nan, 5, np.nan, 7]}, index=index)
    result = causal_impute(frame, limit=2)

    # El faltante inicial no se rellena con el futuro; el hueco de 3 solo se cubre 2 pasos.
    assert result["x"].tolist()[:5] == pytest.approx([np.nan, 1, 1, 1, np.nan], nan_ok=True)
    assert result["x"].iloc[6] == 5  # noqa: PLR2004
    assert result["x_imputado"].tolist() == [0, 0, 1, 1, 0, 0, 1, 0]

    # Cambiar valores futuros no altera ninguna imputación anterior.
    altered = frame.copy()
    altered.iloc[5:, 0] = [-100, np.nan, -200]
    assert causal_impute(altered, limit=2).iloc[:5].equals(result.iloc[:5])


def test_daily_values_are_only_visible_after_publication() -> None:
    days = pd.date_range("2026-06-01", periods=3, freq="D", tz=TZ, name="fecha")
    daily = pd.DataFrame({"aportes_hidricos": [10.0, 20.0, 30.0]}, index=days)
    grid = hourly_grid("2026-06-01", "2026-06-03", TZ)
    aligned = align_daily_causal(daily, grid, lag_days=1)["aportes_hidricos"]

    assert aligned.loc["2026-06-01"].isna().all()  # el día 1 aún no tiene dato publicado
    assert (aligned.loc["2026-06-02"] == 10).all()  # noqa: PLR2004
    assert (aligned.loc["2026-06-03"] == 20).all()  # noqa: PLR2004
    with pytest.raises(ValueError, match=">= 1"):
        align_daily_causal(daily, grid, lag_days=0)


def test_build_tables_aligns_sources_and_marks_gaps(config: dict) -> None:
    grid = hourly_grid("2026-06-01", "2026-06-03", TZ)
    price = pd.DataFrame({"fecha": grid, TARGET: np.arange(len(grid), dtype=float)}).drop(index=[5])
    days = pd.date_range("2026-06-01", periods=3, freq="D", tz=TZ)
    aportes = pd.DataFrame(
        {
            "fecha": days.repeat(2),
            "CodigoSerieHidrologica": ["A", "B"] * 3,
            "aportes_hidricos": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).drop(index=[3])  # el día 2 queda incompleto: falta la serie B
    generation = pd.DataFrame(
        {
            "fecha": days.repeat(2),
            "TipoGeneracion": ["Hidraulica", "Termica"] * 3,
            "generacion_real": [8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
        }
    )
    sources = {"precio_bolsa": price, "aportes_hidricos": aportes, "generacion_real": generation}
    window = {"start": "2026-06-01", "end": "2026-06-03"}
    hourly, daily, report = build_tables(sources, config["simem"]["datasets"], window, config)

    assert hourly.index.equals(grid)
    assert hourly.loc[grid[5], TARGET] == 4 and hourly.loc[grid[5], f"{TARGET}_imputado"] == 1  # noqa: PLR2004
    # Día 3 (horas): recibe el dato del día 2, incompleto, imputado con el del día 1.
    day3 = hourly.loc["2026-06-03"]
    assert (day3["aportes_hidricos"] == 3).all() and (day3["aportes_hidricos_imputado"] == 1).all()  # noqa: PLR2004
    assert (hourly.loc["2026-06-02", "generacion_real_termica"] == 2).all()  # noqa: PLR2004
    assert hourly.loc["2026-06-01", "aportes_hidricos"].isna().all()
    assert report["imputed"][TARGET] == 1
    assert np.isnan(daily.loc["2026-06-01", f"{TARGET}_media"])  # día con una hora faltante
    assert daily.loc["2026-06-02", f"{TARGET}_media"] == pytest.approx(np.mean(range(24, 48)))


def test_missing_configured_source_is_omitted_not_invented(tmp_path, config: dict) -> None:
    price = config["simem"]["datasets"]["precio_bolsa"]
    price.update(revision=None, revision_calendar_id=None)
    config["simem"]["datasets"] = {"precio_bolsa": price}
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        [
            {
                "FechaHora": "2024-04-01 01:00:00",
                "Valor": "123.5",
                "CodigoVariable": "PB_Nal",
                "CodigoDuracion": "PT1H",
                "UnidadMedida": "COP/kWh",
            }
        ]
    ).to_csv(exports / "precio_bolsa.csv", index=False)
    snapshot = ingest(
        config, source_dir=exports, root=tmp_path, start="2024-04-01", end="2024-04-01"
    )

    sources, omitted, window = load_sources(snapshot, ["precio_bolsa", "demanda_nacional"], TARGET)
    assert list(sources) == ["precio_bolsa"] and omitted == ["demanda_nacional"]
    assert window == {"start": "2024-04-01", "end": "2024-04-01"}


# -----------------------------------------------------------------------------
# CARACTERÍSTICAS
# -----------------------------------------------------------------------------
def test_target_features_do_not_look_ahead() -> None:
    frame = hourly_frame(days=10)
    args = {
        "lags": [1, 24, 168],
        "differences": [1, 24],
        "windows": [24, 168],
        "quantiles": [0.1, 0.9],
    }
    base = add_target_features(frame, TARGET, **args)
    cut = 200
    altered = frame.copy()
    altered.iloc[cut + 1 :, 0] *= 50  # se alteran solo los valores posteriores a `cut`
    shifted = add_target_features(altered, TARGET, **args)

    pd.testing.assert_frame_equal(base.iloc[: cut + 1], shifted.iloc[: cut + 1])
    assert base[f"{TARGET}_lag_24"].iloc[cut] == frame[TARGET].iloc[cut - 24]
    assert base[f"{TARGET}_media_24h"].iloc[cut] == pytest.approx(
        frame[TARGET].iloc[cut - 23 : cut + 1].mean()
    )
    assert base[f"{TARGET}_lag_168"].iloc[:168].isna().all()


def test_cyclical_encoding_wraps_periods() -> None:
    index = pd.date_range("2026-06-01", periods=48, freq="h", tz=TZ)
    encoded = add_cyclical_features(pd.DataFrame(index=index), ["hour", "dayofweek"])
    assert np.allclose(encoded["hour_sin"] ** 2 + encoded["hour_cos"] ** 2, 1)
    hour = encoded[["hour_sin", "hour_cos"]]
    assert np.allclose(hour.iloc[0], hour.iloc[24])  # 00:00 de dos días consecutivos
    assert np.allclose(hour.iloc[23], hour.iloc[0], atol=0.3)  # 23:00 queda junto a 00:00
    assert not np.allclose(encoded["dayofweek_sin"].iloc[0], encoded["dayofweek_sin"].iloc[24])
    with pytest.raises(ValueError, match="no soportados"):
        add_cyclical_features(pd.DataFrame(index=index), ["minute"])


# -----------------------------------------------------------------------------
# PARTICIÓN, ESCALADO Y VENTANAS
# -----------------------------------------------------------------------------
def test_chronological_split_is_ordered_disjoint_and_complete() -> None:
    splits = chronological_split(1000, 0.7, 0.15, 0.15)
    assert [(s.start, s.stop) for s in splits.values()] == [(0, 700), (700, 850), (850, 1000)]
    with pytest.raises(ValueError, match="inválidas"):
        chronological_split(1000, 0.7, 0.2, 0.2)
    with pytest.raises(ValueError, match="suficientes"):
        chronological_split(5, 0.7, 0.15, 0.15)


def test_windows_never_cross_partition_boundaries() -> None:
    n, n_features = 100, 3
    features = np.arange(n * n_features, dtype=float).reshape(n, n_features)
    target = np.arange(n, dtype=float)
    part = slice(40, 70)
    x, y, ends = make_windows(features, target, part, LOOKBACK, HORIZON)

    assert x.shape == (30 - LOOKBACK - HORIZON + 1, LOOKBACK, n_features)
    assert y.shape == (len(x), HORIZON)
    first_input, last_target = ends - LOOKBACK + 1, ends + HORIZON
    assert first_input.min() == part.start and last_target.max() == part.stop - 1
    assert np.array_equal(x[:, -1, 0], features[ends, 0])  # la entrada termina en t
    assert np.array_equal(y[:, 0], target[ends + 1])  # el objetivo empieza en t + 1

    target_with_gap = target.copy()
    target_with_gap[55] = np.nan
    _, _, kept = make_windows(features, target_with_gap, part, LOOKBACK, HORIZON)
    assert not np.isin(55, np.concatenate([np.arange(t + 1, t + HORIZON + 1) for t in kept]))


def test_scaler_is_fit_only_on_training_rows(config: dict) -> None:
    hourly = hourly_frame(days=30)
    splits = chronological_split(len(hourly), 0.7, 0.15, 0.15)
    # Validación y prueba en otro régimen de precios: si se filtraran, moverían la mediana.
    hourly.iloc[splits["train"].stop :, 0] += 1e5
    result = prepare_datasets(hourly, config)
    scalers = result["scalers"]
    train = result["table"].iloc[splits["train"]]

    assert scalers["fit_end"] == str(hourly.index[splits["train"].stop - 1])
    assert scalers["target"].center_[0] == pytest.approx(train[TARGET].median())
    assert scalers["target"].center_[0] < hourly[TARGET].median()
    columns = scalers["feature_columns"]
    assert np.allclose(scalers["features"].center_, train[columns].median())
    assert not any(c.endswith(("_sin", "_cos", "_imputado")) for c in columns)


def test_prepared_windows_stay_inside_each_partition(config: dict) -> None:
    hourly = hourly_frame(days=30)
    result = prepare_datasets(hourly, config)
    for name, part in result["splits"].items():
        x, y, ends = result["windows"][name]
        assert len(x) > 0 and x.dtype == np.float32
        assert x.shape[1:] == (LOOKBACK, result["table"].shape[1]) and y.shape[1] == HORIZON
        assert (ends - LOOKBACK + 1 >= part.start).all() and (ends + HORIZON < part.stop).all()
    # Las particiones no comparten ningún instante objetivo.
    targets = {
        name: set(np.concatenate([ends + h for h in range(1, HORIZON + 1)]))
        for name, (_, _, ends) in result["windows"].items()
    }
    assert not targets["train"] & targets["validation"]
    assert not targets["validation"] & targets["test"]


def test_run_writes_tensors_metadata_and_scaler(tmp_path, config: dict) -> None:
    interim = tmp_path / config["paths"]["data"]["interim"]
    interim.mkdir(parents=True)
    hourly_frame(days=30).rename_axis("fecha").reset_index().to_parquet(
        interim / "simem_hourly.parquet", index=False
    )
    processed = run(config, root=tmp_path)

    train = np.load(processed / "train.npz")
    assert train["X"].ndim == 3 and train["X"].shape[1] == LOOKBACK  # noqa: PLR2004
    assert train["y"].shape == (len(train["X"]), HORIZON)
    bundle = joblib.load(tmp_path / config["paths"]["models"]["trained"] / "scaler.pkl")
    restored = bundle["target"].inverse_transform(train["y"][:, :1])
    assert restored.min() > 0  # vuelve a la escala original en COP/kWh
    assert (processed / "feature_metadata.json").is_file()
