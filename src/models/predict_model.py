# -*- coding: utf-8 -*-  # noqa: UP009
"""Evaluación fuera de muestra del modelo campeón en COP/kWh (Sprint 4).

Carga el campeón exportado por ``src.models.optimize`` y el conjunto de prueba
``data/processed/test.npz``, que ninguna etapa anterior usó para entrenar ni para elegir
modelos. Invierte el escalado con ``models/trained/scaler.pkl`` y compara contra:

- persistencias estacionales de 24 h y 168 h (``evaluation.naive_seasons``): el precio de la
  misma hora de ayer o de la semana pasada;
- el baseline Ridge del Sprint 3, si su archivo existe.

Métricas por modelo, sobre todas las horas del horizonte:

- **MAE** y **RMSE** en COP/kWh.
- **MAPE acotado**: ``|e| / max(|y|, piso)``. El precio de bolsa puede acercarse a cero (o ser
  negativo en otros mercados); con el MAPE clásico una sola hora así dispara el promedio. El
  piso (``evaluation.mape_floor_cop``) solo actúa en esas horas y deja intactas las demás.
- **sMAPE**: ``2|e| / (|y| + |ŷ|)``, simétrico y acotado en [0 %, 200 %].
- **Acierto direccional**: si el precio sube o baja respecto a la hora anterior. Para la primera
  hora del horizonte, la referencia es el último precio observado; ver ``direction_labels``.
- **Mejora relativa** del MAE frente a cada baseline: ``1 - MAE_modelo / MAE_baseline``.

Salidas en ``reports/metrics/``:

- ``test_metrics.csv``: una fila por modelo.
- ``test_metrics_by_horizon.csv``: MAE por hora de anticipación (1 a 24 h).
- ``test_predictions.parquet``: pronósticos en formato largo para figuras y notebook.
- ``test_summary.json``: contexto (periodo, muestras, campeón) y matrices de confusión.

Ejemplo:
    $ python -m src.models.predict_model

Proyecto: deep-learning-ws

Programa: predict_model.py

version: 1.0.0

lenguaje: Python 3.14.5

CD: 20261125

LUD: 20261125

Comentarios:
    - 2026-11-25 Sebastian Sanchez -> creación.
"""

__authors__ = ["Sebastian Sanchez"]
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Sebastian Sanchez"]
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-25"

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.features.build_features import METADATA_FILE
from src.models.architectures import seasonal_naive
from src.models.optimize import load_champion
from src.models.train_model import (
    BASELINE_FILE,
    Split,
    feature_target_to_cop,
    load_split,
    to_cop,
)
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
CHAMPION = "champion"
METRICS_FILE = "test_metrics.csv"
HORIZON_FILE = "test_metrics_by_horizon.csv"
PREDICTIONS_FILE = "test_predictions.parquet"
SUMMARY_FILE = "test_summary.json"
UP, DOWN = "sube", "baja"

logger = logging.getLogger(__name__)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# MÉTRICAS
# -----------------------------------------------------------------------------
def bounded_mape(y_true: np.ndarray, y_pred: np.ndarray, floor: float) -> float:
    """MAPE en % con denominador ``max(|y|, floor)``.

    Raises:
        ValueError: Si ``floor`` no es positivo.
    """
    if floor <= 0:
        raise ValueError(f"El piso del MAPE debe ser positivo: {floor}")
    denominator = np.maximum(np.abs(y_true), floor)
    return float(100 * np.mean(np.abs(y_pred - y_true) / denominator))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE en %, con 0 cuando real y pronóstico son ambos cero."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(100 * np.mean(ratio))


def direction_labels(values: np.ndarray, last_observed: np.ndarray) -> np.ndarray:
    """Signo del cambio de cada hora frente a la anterior: ``+1`` sube, ``-1`` baja, ``0`` igual.

    Args:
        values: Trayectoria ``(B, H)`` (real o pronosticada) en COP/kWh.
        last_observed: Último precio conocido al pronosticar, ``(B,)``. Es la referencia de la
            primera hora, tanto para la trayectoria real como para la pronosticada.
    """
    path = np.concatenate([last_observed[:, None], values], axis=1)
    return np.sign(np.diff(path, axis=1))


def directional_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, last_observed: np.ndarray
) -> tuple[float, pd.DataFrame]:
    """Tasa de acierto en la dirección del precio y matriz de confusión.

    Las horas en que el precio real no cambia no tienen dirección y se excluyen. Un pronóstico
    plano cuenta como error, porque no anticipa ni la subida ni la bajada.

    Returns:
        Tupla ``(acierto en %, matriz de confusión (filas: real; columnas: pronóstico))``.
    """
    actual = direction_labels(y_true, last_observed).ravel()
    predicted = direction_labels(y_pred, last_observed).ravel()
    moving = actual != 0
    accuracy = float(100 * np.mean(actual[moving] == predicted[moving])) if moving.any() else np.nan
    names = {1: UP, -1: DOWN, 0: "igual"}
    matrix = pd.crosstab(
        pd.Series([names[v] for v in actual[moving]], name="real"),
        pd.Series([names[v] for v in predicted[moving]], name="pronóstico"),
    ).reindex(index=[UP, DOWN], columns=[UP, DOWN, "igual"], fill_value=0)
    return accuracy, matrix


def model_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, last_observed: np.ndarray, floor: float
) -> tuple[dict[str, float], pd.DataFrame]:
    """Todas las métricas de un modelo sobre la partición de prueba."""
    error = y_pred - y_true
    accuracy, matrix = directional_metrics(y_true, y_pred, last_observed)
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_acotado": bounded_mape(y_true, y_pred, floor),
        "smape": smape(y_true, y_pred),
        "acierto_direccional": accuracy,
    }
    return metrics, matrix


# -----------------------------------------------------------------------------
# PRONÓSTICOS
# -----------------------------------------------------------------------------
@dataclass
class Forecasts:
    """Pronósticos de prueba en COP/kWh, todos con forma ``(B, H)``."""

    y_true: np.ndarray
    last_observed: np.ndarray
    models: dict[str, np.ndarray]
    input_end: pd.DatetimeIndex


def load_test(config: dict, root: Path) -> tuple[Split, dict, dict]:
    """Lee la partición de prueba, los metadatos de características y el escalador."""
    paths = config["paths"]
    processed = root / paths["data"]["processed"]
    metadata = json.loads((processed / METADATA_FILE).read_text(encoding="utf-8"))
    scalers = joblib.load(root / paths["models"]["trained"] / config["features"]["scaler_file"])
    return load_split(processed, "test"), metadata, scalers


def build_forecasts(config: dict, *, root: Path = PROJECT_PATH) -> tuple[Forecasts, dict]:
    """Pronostica la partición de prueba con el campeón y los baselines.

    Returns:
        Tupla ``(pronósticos, metadatos del campeón)``.
    """
    test, metadata, scalers = load_test(config, root)
    target = metadata["target"]
    target_index = metadata["feature_columns"].index(target)
    forecast, champion_meta = load_champion(root / config["paths"]["models"]["trained"])

    history = feature_target_to_cop(test.x[:, :, target_index], scalers, target)
    horizon = test.y.shape[1]
    models = {CHAMPION: to_cop(forecast(test.x), scalers)}
    for season in config["evaluation"]["naive_seasons"]:
        if season > history.shape[1]:
            logger.warning(
                "La ventana (%d h) no cubre la persistencia de %d h", history.shape[1], season
            )
            continue
        models[f"naive_{season}h"] = seasonal_naive(history, horizon, season)

    ridge_path = root / config["paths"]["models"]["checkpoints"] / BASELINE_FILE
    if ridge_path.is_file():
        ridge = joblib.load(ridge_path)
        models["ridge"] = to_cop(ridge.predict(test.x)[..., 0], scalers)
    else:
        logger.warning("No existe %s: se omite el baseline Ridge", ridge_path)

    input_end = pd.DatetimeIndex(pd.to_datetime(test.input_end)).tz_convert(
        config["data"]["timezone"]
    )
    forecasts = Forecasts(to_cop(test.y, scalers), history[:, -1], models, input_end)
    return forecasts, champion_meta


def metrics_table(forecasts: Forecasts, floor: float) -> tuple[pd.DataFrame, dict]:
    """Tabla de métricas por modelo, con la mejora relativa del MAE frente a cada baseline.

    Returns:
        Tupla ``(tabla, matrices de confusión por modelo como diccionarios)``.
    """
    rows, matrices = {}, {}
    for name, prediction in forecasts.models.items():
        rows[name], matrix = model_metrics(
            forecasts.y_true, prediction, forecasts.last_observed, floor
        )
        matrices[name] = matrix.to_dict(orient="index")
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("modelo")
    for baseline in (name for name in table.index if name != CHAMPION):
        table[f"mejora_mae_vs_{baseline}"] = 1 - table["mae"] / table.loc[baseline, "mae"]
    return table, matrices


def horizon_table(forecasts: Forecasts) -> pd.DataFrame:
    """MAE por hora de anticipación (1 = la hora siguiente al corte)."""
    return pd.DataFrame(
        {
            name: np.mean(np.abs(prediction - forecasts.y_true), axis=0)
            for name, prediction in forecasts.models.items()
        },
        index=pd.RangeIndex(1, forecasts.y_true.shape[1] + 1, name="horizonte_h"),
    )


def predictions_frame(forecasts: Forecasts) -> pd.DataFrame:
    """Pronósticos en formato largo: una fila por ventana y hora del horizonte."""
    n_windows, horizon = forecasts.y_true.shape
    steps = np.tile(np.arange(1, horizon + 1), n_windows)
    input_end = np.repeat(forecasts.input_end, horizon)
    frame = pd.DataFrame(
        {
            "input_end": input_end,
            "horizonte_h": steps,
            "fecha_objetivo": input_end + pd.to_timedelta(steps, unit="h"),
            "real": forecasts.y_true.ravel(),
            "ultimo_observado": np.repeat(forecasts.last_observed, horizon),
        }
    )
    for name, prediction in forecasts.models.items():
        frame[name] = prediction.ravel()
    return frame


def run(config: dict, *, root: Path = PROJECT_PATH) -> dict:
    """Evalúa el campeón en prueba y escribe métricas y pronósticos en ``reports/metrics/``.

    Returns:
        Resumen con el contexto de la evaluación y las métricas por modelo.
    """
    evaluation = config["evaluation"]
    forecasts, champion_meta = build_forecasts(config, root=root)
    table, matrices = metrics_table(forecasts, evaluation["mape_floor_cop"])

    output = root / config["paths"]["reports"]["metrics"]
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / METRICS_FILE, float_format="%.4f")
    horizon_table(forecasts).to_csv(output / HORIZON_FILE, float_format="%.4f")
    predictions_frame(forecasts).to_parquet(output / PREDICTIONS_FILE, index=False)

    horizon = int(forecasts.y_true.shape[1])
    first, last = forecasts.input_end.min(), forecasts.input_end.max()
    summary = {
        "champion": {
            k: champion_meta[k] for k in ("architecture", "params", "lookback", "val_cop")
        },
        "test": {
            "windows": len(forecasts.y_true),
            "hours_evaluated": int(forecasts.y_true.size),
            "horizon_h": horizon,
            "first_forecast_hour": (first + np.timedelta64(1, "h")).isoformat(),
            "last_forecast_hour": (last + np.timedelta64(horizon, "h")).isoformat(),
        },
        "mape_floor_cop": evaluation["mape_floor_cop"],
        "hours_below_mape_floor": int(
            np.sum(np.abs(forecasts.y_true) < evaluation["mape_floor_cop"])
        ),
        "metrics": json.loads(table.to_json(orient="index")),
        "confusion_matrices": matrices,
    }
    (output / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_event(
        logging.INFO, "evaluation_finished", path=output, champion=champion_meta["architecture"]
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos de la evaluación."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        summary = run(read_config_mapping(args.config))
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo en la evaluación: %s", exc)
        log_event(logging.ERROR, "evaluation_failed", error=str(exc))
        return 1
    test = summary["test"]
    logger.info(
        "Prueba: %d ventanas, %s a %s",
        test["windows"],
        test["first_forecast_hour"],
        test["last_forecast_hour"],
    )
    for name, row in summary["metrics"].items():
        logger.info(
            "%-12s MAE %7.2f  RMSE %7.2f  MAPE %6.2f%%  sMAPE %6.2f%%  dirección %5.1f%%",
            name,
            row["mae"],
            row["rmse"],
            row["mape_acotado"],
            row["smape"],
            row["acierto_direccional"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
