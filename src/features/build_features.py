# -*- coding: utf-8 -*-  # noqa: UP009
"""Ingeniería de características y tensores secuenciales sin fuga de información (Sprint 2).

Parte de la tabla horaria de ``data/interim/`` (ver ``src.data.preprocess``) y produce:

1. Características causales: cada fila ``t`` usa solo información disponible hasta ``t``.
   Incluye rezagos, diferencias y ventanas móviles del precio, codificación cíclica del
   calendario e indicadores del dominio energético cuando sus columnas existen.
2. Partición cronológica estricta en entrenamiento, validación y prueba.
3. Escalado ajustado únicamente con la partición de entrenamiento. Se serializa en
   ``models/trained/scaler.pkl`` para la inferencia y para invertir las predicciones.
4. Ventanas deslizantes ``(muestras, pasos_tiempo, características)`` con objetivos
   ``(muestras, horizonte)``. Ninguna ventana cruza la frontera de su partición: tanto la
   entrada como el objetivo caen completos dentro de ella.

Ejemplo:
    $ python -m src.features.build_features

Proyecto: deep-learning-ws

Programa: build_features.py

Soporte: leidymilenahernandezc@gmail.com

version: 1.0.0

lenguaje: Python 3.14.5

CD: 20261123

LUD: 20261123

Comentarios:
    - 2026-11-23 Milena -> creación.
"""

__authors__ = ["Milena"]
__contact__ = "leidymilenahernandezc@gmail.com"
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Milena"]
__email__ = "leidymilenahernandezc@gmail.com"
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-23"

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.data.preprocess import HOURLY_FILE, IMPUTED_SUFFIX
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
SPLITS = ("train", "validation", "test")
SCALERS = {"robust": RobustScaler, "standard": StandardScaler}
# Columnas acotadas o binarias que no se escalan.
UNSCALED_SUFFIXES = (IMPUTED_SUFFIX, "_sin", "_cos")
METADATA_FILE = "feature_metadata.json"
HOURS_PER_DAY = 24

logger = logging.getLogger(__name__)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _console_handler.setFormatter(_console_formatter)
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# CARACTERÍSTICAS
# -----------------------------------------------------------------------------
def add_target_features(  # noqa: PLR0913
    frame: pd.DataFrame,
    target: str,
    *,
    lags: list[int],
    differences: list[int],
    windows: list[int],
    quantiles: list[float],
) -> pd.DataFrame:
    """Agrega rezagos, diferencias y estadísticas móviles del objetivo.

    Los rezagos salen del EDA (notebook 01, ACF/PACF). El precio tiene una persistencia fuerte
    de una hora (ACF(1) = 0,86), picos diarios en 24 y 48 h y un pico semanal en 168 h, que
    persiste en la serie diferenciada. El rezago de 168 h amplía la memoria más allá de la
    ventana de entrada. Las diferencias responden a que la serie es I(1) en nivel. Las ventanas
    móviles son retrospectivas, de ``t - w + 1`` a ``t``, y exigen la ventana completa
    (``min_periods = w``): una ventana incompleta o con faltantes produce ``NaN``.

    Args:
        frame: Tabla horaria en orden cronológico.
        target: Columna del precio.
        lags: Rezagos en horas.
        differences: Órdenes de diferencia en horas.
        windows: Tamaños de ventana móvil en horas.
        quantiles: Cuantiles móviles en ``(0, 1)``.

    Returns:
        Copia de ``frame`` con las nuevas columnas.
    """
    y = frame[target]
    new = {f"{target}_lag_{lag}": y.shift(lag) for lag in lags}
    new |= {f"{target}_diff_{d}": y.diff(d) for d in differences}
    for w in windows:
        rolling = y.rolling(w, min_periods=w)
        new[f"{target}_media_{w}h"] = rolling.mean()
        new[f"{target}_std_{w}h"] = rolling.std()
        for q in quantiles:
            new[f"{target}_p{round(q * 100):02d}_{w}h"] = rolling.quantile(q)
    return frame.assign(**new)


def add_cyclical_features(frame: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    """Codifica hora, día de la semana y mes con seno y coseno.

    Así la hora 23 queda junto a la 0 y diciembre junto a enero, cosa que un entero no logra.

    Raises:
        ValueError: Si se pide un componente no soportado.
    """
    index = pd.DatetimeIndex(frame.index)
    cycles = {
        "hour": (index.hour, 24),
        "dayofweek": (index.dayofweek, 7),
        "month": (index.month - 1, 12),
    }
    if unknown := set(components) - set(cycles):
        raise ValueError(f"Componentes cíclicos no soportados: {sorted(unknown)}")
    new = {}
    for name in components:
        value, period = cycles[name]
        angle = 2 * np.pi * np.asarray(value) / period
        new[f"{name}_sin"] = np.sin(angle)
        new[f"{name}_cos"] = np.cos(angle)
    return frame.assign(**new)


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores del mercado eléctrico si sus columnas existen.

    - ``brecha_demanda_aportes``: demanda de las últimas 24 h menos los aportes hídricos del
      último día publicado. Ambas fuentes de SIMEM están en energía (kWh). Una brecha positiva
      indica que el agua que llega no cubre el consumo y que se necesita más generación térmica,
      que es más cara.
    - ``participacion_termica``: fracción térmica de la generación hidráulica + térmica del
      último día publicado.

    Si falta alguna columna, el indicador se omite y la omisión queda registrada.
    """
    new = {}
    if {"demanda_nacional", "aportes_hidricos"} <= set(frame.columns):
        demand_24h = frame["demanda_nacional"].rolling(HOURS_PER_DAY, min_periods=HOURS_PER_DAY)
        new["brecha_demanda_aportes"] = demand_24h.sum() - frame["aportes_hidricos"]
    else:
        logger.warning("Sin demanda_nacional o aportes_hidricos: se omite brecha_demanda_aportes")

    thermal, hydro = "generacion_real_termica", "generacion_real_hidraulica"
    if {thermal, hydro} <= set(frame.columns):
        total = frame[thermal] + frame[hydro]
        new["participacion_termica"] = frame[thermal] / total.where(total > 0)
    else:
        logger.warning("Sin generación térmica e hidráulica: se omite participacion_termica")
    return frame.assign(**new)


def build_feature_table(hourly: pd.DataFrame, target: str, params: dict) -> pd.DataFrame:
    """Aplica todas las transformaciones causales sobre la tabla horaria."""
    table = add_target_features(
        hourly,
        target,
        lags=params["lags"],
        differences=params["differences"],
        windows=params["rolling_windows"],
        quantiles=params["rolling_quantiles"],
    )
    table = add_domain_features(table)
    return add_cyclical_features(table, params["cyclical"])


# -----------------------------------------------------------------------------
# PARTICIÓN, ESCALADO Y VENTANAS
# -----------------------------------------------------------------------------
def chronological_split(n: int, train: float, validation: float, test: float) -> dict[str, slice]:
    """Divide ``n`` pasos consecutivos en tres bloques cronológicos disjuntos.

    Raises:
        ValueError: Si las fracciones no son positivas, no suman 1 o dejan un bloque vacío.
    """
    fractions = (train, validation, test)
    if min(fractions) <= 0 or not math.isclose(sum(fractions), 1.0):
        raise ValueError(f"Fracciones de partición inválidas: {fractions}")
    n_train, n_val = int(n * train), int(n * validation)
    if min(n_train, n_val, n - n_train - n_val) <= 0:
        raise ValueError(f"No hay suficientes pasos ({n}) para las tres particiones")
    return {
        "train": slice(0, n_train),
        "validation": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }


def scaled_columns(columns: pd.Index | list[str]) -> list[str]:
    """Columnas continuas que se escalan; las cíclicas y los indicadores 0/1 se conservan."""
    return [c for c in columns if not str(c).endswith(UNSCALED_SUFFIXES)]


def fit_scalers(train: pd.DataFrame, target: str, kind: str) -> dict:
    """Ajusta los escaladores con la partición de entrenamiento únicamente.

    Los faltantes se ignoran al ajustar, como hace scikit-learn.

    Args:
        train: Filas de entrenamiento de la tabla de características.
        target: Columna objetivo.
        kind: ``"robust"`` o ``"standard"``.

    Returns:
        Paquete serializable con el escalador de características, el del objetivo (para
        invertir las predicciones a COP/kWh) y el rango temporal usado para ajustarlos.
    """
    if kind not in SCALERS:
        raise ValueError(f"Escalador no soportado: {kind}; usa {sorted(SCALERS)}")
    columns = scaled_columns(train.columns)
    return {
        "kind": kind,
        "feature_columns": columns,
        "features": SCALERS[kind]().fit(train[columns]),
        "target_column": target,
        "target": SCALERS[kind]().fit(train[[target]]),
        "fit_start": str(train.index[0]),
        "fit_end": str(train.index[-1]),
    }


def apply_scalers(table: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Escala características y objetivo con un paquete ya ajustado."""
    columns = bundle["feature_columns"]
    features = table.copy()
    features[columns] = bundle["features"].transform(table[columns])
    target = bundle["target"].transform(table[[bundle["target_column"]]])[:, 0]
    return features, pd.Series(target, index=table.index, name=bundle["target_column"])


def make_windows(
    features: np.ndarray, target: np.ndarray, part: slice, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye ventanas deslizantes dentro de una única partición.

    Para cada instante ``t`` de la partición, la entrada es ``features[t - lookback + 1 .. t]``
    y el objetivo es ``target[t + 1 .. t + horizon]``. Solo se generan las ventanas en las que
    ambos rangos están completos dentro de ``part``. Se descartan las ventanas con algún
    ``NaN``, por ejemplo durante el calentamiento de los rezagos o en huecos no imputados.

    Args:
        features: Matriz ``(pasos, características)`` en orden cronológico.
        target: Vector ``(pasos,)`` del objetivo.
        part: Rango ``[start, stop)`` de la partición.
        lookback: Pasos de entrada por muestra.
        horizon: Pasos de pronóstico por muestra.

    Returns:
        Tupla ``(X, y, t)`` con ``X`` de forma ``(muestras, lookback, características)``,
        ``y`` de forma ``(muestras, horizon)`` y ``t`` con el índice absoluto del último
        paso de entrada de cada muestra.
    """
    ends = np.arange(part.start + lookback - 1, part.stop - horizon)
    n_features = features.shape[1]
    if ends.size == 0:
        empty = np.empty((0, lookback, n_features), dtype=np.float32)
        return empty, np.empty((0, horizon), dtype=np.float32), ends
    x = sliding_window_view(features, lookback, axis=0).transpose(0, 2, 1)[ends - lookback + 1]
    y = sliding_window_view(target, horizon)[ends + 1]
    valid = ~(np.isnan(x).any(axis=(1, 2)) | np.isnan(y).any(axis=1))
    return x[valid].astype(np.float32), y[valid].astype(np.float32), ends[valid]


def prepare_datasets(hourly: pd.DataFrame, config: dict) -> dict:
    """Ejecuta características, partición, escalado y ventanas en memoria.

    Args:
        hourly: Tabla horaria de ``data/interim/`` indexada por ``fecha``.
        config: Configuración completa (usa ``data``, ``features`` y ``training``).

    Returns:
        Diccionario con ``table`` (características sin escalar), ``splits``, ``scalers``
        y ``windows`` (``{partición: (X, y, t)}``).

    Raises:
        ValueError: Si alguna partición queda sin ventanas válidas.
    """
    target, params, training = (
        config["data"]["target_column"],
        config["features"],
        config["training"],
    )
    if not hourly.index.is_monotonic_increasing or not hourly.index.is_unique:
        raise ValueError("La tabla horaria debe estar ordenada y sin marcas repetidas")

    table = build_feature_table(hourly, target, params)
    splits = chronological_split(
        len(table),
        training["train_fraction"],
        training["validation_fraction"],
        training["test_fraction"],
    )
    scalers = fit_scalers(table.iloc[splits["train"]], target, params["scaler"])
    features, scaled_target = apply_scalers(table, scalers)

    matrix, vector = features.to_numpy(np.float64), scaled_target.to_numpy(np.float64)
    lookback, horizon = training["lookback_steps"], training["forecast_horizon"]
    windows = {
        name: make_windows(matrix, vector, part, lookback, horizon) for name, part in splits.items()
    }
    if empty := [name for name, (x, _, _) in windows.items() if not len(x)]:
        raise ValueError(
            f"Particiones sin ventanas válidas {empty}: amplía la ventana de datos o reduce "
            f"lookback_steps ({lookback}) / forecast_horizon ({horizon})"
        )
    return {"table": table, "splits": splits, "scalers": scalers, "windows": windows}


def run(config: dict, *, root: Path = PROJECT_PATH) -> Path:
    """Lee ``data/interim/``, prepara los tensores y los guarda en ``data/processed/``.

    Escribe ``train.npz``, ``validation.npz`` y ``test.npz`` (claves ``X``, ``y`` y
    ``input_end``), ``feature_metadata.json`` y el escalador en ``models/trained/``.

    Returns:
        Directorio de salida ``data/processed/``.
    """
    paths = config["paths"]
    interim_path = root / paths["data"]["interim"] / HOURLY_FILE
    if not interim_path.is_file():
        raise FileNotFoundError(
            f"No existe {interim_path}; ejecuta primero `python -m src.data.preprocess`."
        )
    hourly = pd.read_parquet(interim_path).set_index("fecha")
    result = prepare_datasets(hourly, config)
    table, index = result["table"], result["table"].index

    processed = root / paths["data"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (x, y, ends) in result["windows"].items():
        part = result["splits"][name]
        np.savez_compressed(
            processed / f"{name}.npz",
            X=x,
            y=y,
            input_end=index[ends].strftime("%Y-%m-%dT%H:%M:%S%z").to_numpy(str),
        )
        summary[name] = {
            "start": str(index[part.start]),
            "end": str(index[part.stop - 1]),
            "steps": part.stop - part.start,
            "samples": len(x),
            "X_shape": list(x.shape),
            "y_shape": list(y.shape),
        }
        logger.info("%s: X%s y%s", name, x.shape, y.shape)

    trained = root / paths["models"]["trained"]
    trained.mkdir(parents=True, exist_ok=True)
    scaler_path = trained / config["features"]["scaler_file"]
    joblib.dump(result["scalers"], scaler_path)

    metadata = {
        "source": str(interim_path),
        "target": config["data"]["target_column"],
        "feature_columns": list(table.columns),
        "scaled_columns": result["scalers"]["feature_columns"],
        "lookback_steps": config["training"]["lookback_steps"],
        "forecast_horizon": config["training"]["forecast_horizon"],
        "scaler": {"path": str(scaler_path), "kind": result["scalers"]["kind"]},
        "splits": summary,
    }
    (processed / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_event(logging.INFO, "features_published", path=processed, splits=summary)
    return processed


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos de la ingeniería de características."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        run(read_config_mapping(args.config))
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo en la ingeniería de características: %s", exc)
        log_event(logging.ERROR, "features_failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
