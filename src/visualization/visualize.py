# -*- coding: utf-8 -*-  # noqa: UP009
"""Figuras de publicación e interpretabilidad del modelo campeón (Sprint 4).

Lee los pronósticos de prueba que escribe ``src.models.predict_model`` en ``reports/metrics/`` y
genera figuras a ``evaluation.dpi`` (300 DPI) en ``reports/figures/evaluacion/``:

1. ``serie_real_vs_pronostico.png``: precio real contra el campeón y la persistencia de 24 h en
   modo *day-ahead* (cortes a las 23:00, que pronostican el día siguiente completo, como en el
   mercado). Incluye un zoom de 72 h sobre el episodio de mayor volatilidad.
2. ``residuales.png``: serie temporal de los errores y su distribución.
3. ``error_por_hora.png``: distribución del error absoluto según la hora del día.
4. ``mae_por_horizonte.png``: cómo crece el error con la anticipación, por modelo.
5. ``importancia_variables.png`` e ``importancia_temporal.png``: atribuciones por gradientes
   integrados (Sundararajan et al., 2017) agregadas por variable y por hora de la ventana.

**Gradientes integrados.** Para cada ventana ``x`` se integra el gradiente del pronóstico medio
de las 24 h a lo largo del camino recto desde una referencia ``x'`` hasta ``x``:
``IG_i = (x_i - x'_i) · ∫ ∂f(x' + a·(x - x')) / ∂x_i da`, con `a` de 0 a 1`. La referencia es la ventana de ceros,
que en la escala robusta equivale a la mediana de entrenamiento de cada variable: la atribución
mide cuánto mueve el pronóstico que una variable se aleje de su valor típico. Las atribuciones
cumplen ``Σ IG = f(x) - f(x')`` (completitud); el error de esa identidad se registra como control
de la aproximación numérica de la integral. La importancia global de una variable es la media
del valor absoluto de sus atribuciones, que se reporta también en COP/kWh multiplicando por la
escala del objetivo.

Ejemplo:
    $ python -m src.visualization.visualize

Proyecto: deep-learning-ws

Programa: visualize.py

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
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.models.architectures import MEDIAN
from src.models.optimize import CHAMPION_META_FILE
from src.models.predict_model import (
    CHAMPION,
    HORIZON_FILE,
    PREDICTIONS_FILE,
    load_test,
)
from src.models.train_model import load_network
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
IMPORTANCE_FILE = "importancia_gradientes_integrados.csv"
DAY_AHEAD_CUT_HOUR = 23
ZOOM_HOURS = 72
TOP_FEATURES = 15
COLORS = {
    "real": "#1b1b1b",
    CHAMPION: "#1f5aa6",
    "naive_24h": "#d9822b",
    "naive_168h": "#8c8c8c",
    "ridge": "#5b9b4a",
}
LABELS = {
    "real": "Real",
    CHAMPION: "Campeón",
    "naive_24h": "Persistencia 24 h",
    "naive_168h": "Persistencia 168 h",
    "ridge": "Ridge",
}
UNIT = "COP/kWh"

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

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------
def save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Guarda la figura con fondo blanco y la cierra."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log_event(logging.INFO, "figure_exported", path=path)
    return path


def day_ahead(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pronósticos sin solapamiento, una hora objetivo por fila.

    Usa los cortes de las 23:00 (pronóstico del día siguiente completo). Si no hay ninguno, por
    ejemplo en una partición muy corta, usa el pronóstico a una hora de cada ventana.
    """
    cuts = predictions[predictions["input_end"].dt.hour == DAY_AHEAD_CUT_HOUR]
    if cuts.empty:
        cuts = predictions[predictions["horizonte_h"] == 1]
    return cuts.set_index("fecha_objetivo").sort_index()


def most_volatile_window(series: pd.Series, hours: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intervalo de ``hours`` horas con la mayor desviación estándar del precio real."""
    if len(series) <= hours:
        return series.index[0], series.index[-1]
    rolling = series.rolling(hours).std()
    end = rolling.idxmax()
    return end - np.timedelta64(hours - 1, "h"), end


def baselines(frame: pd.DataFrame) -> list[str]:
    """Columnas de baselines presentes, en orden de presentación."""
    return [name for name in ("naive_24h", "naive_168h", "ridge") if name in frame.columns]


# -----------------------------------------------------------------------------
# FIGURAS DE ERROR
# -----------------------------------------------------------------------------
def plot_forecast(predictions: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Serie real contra pronóstico, con zoom en el episodio más volátil."""
    series = day_ahead(predictions)
    start, end = most_volatile_window(series["real"], ZOOM_HOURS)
    fig, (full, zoom) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [1.3, 1]}
    )
    for ax, data in ((full, series), (zoom, series.loc[start:end])):
        ax.plot(data.index, data["real"], color=COLORS["real"], lw=1.2, label=LABELS["real"])
        ax.plot(data.index, data[CHAMPION], color=COLORS[CHAMPION], lw=1.1, label=LABELS[CHAMPION])
        if "naive_24h" in data:
            ax.plot(
                data.index,
                data["naive_24h"],
                color=COLORS["naive_24h"],
                lw=0.9,
                ls="--",
                label=LABELS["naive_24h"],
            )
        ax.set_ylabel(f"Precio de bolsa ({UNIT})")
    full.axvspan(start, end, color=COLORS[CHAMPION], alpha=0.08, lw=0)
    full.set_title("Conjunto de prueba: pronóstico del día siguiente (cortes a las 23:00)")
    zoom.set_title(f"Zoom: {ZOOM_HOURS} h de mayor volatilidad del precio real")
    handles, labels = full.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    full.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    fig.tight_layout()
    return save(fig, path, dpi)


def plot_residuals(predictions: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Serie temporal de residuales (day-ahead) e histograma de todos los horizontes."""
    series = day_ahead(predictions)
    residual_series = series[CHAMPION] - series["real"]
    residuals = predictions[CHAMPION] - predictions["real"]
    fig, (line, hist) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    line.axhline(0, color=COLORS["real"], lw=0.8)
    line.plot(residual_series.index, residual_series, color=COLORS[CHAMPION], lw=0.9)
    line.set_title("Residual del pronóstico del día siguiente")
    line.set_ylabel(f"Pronóstico - real ({UNIT})")
    line.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    hist.hist(residuals, bins=40, color=COLORS[CHAMPION], alpha=0.85)
    hist.axvline(residuals.mean(), color=COLORS["naive_24h"], lw=1.2, label="media")
    hist.set_title("Distribución (todos los horizontes)")
    hist.set_xlabel(f"Pronóstico - real ({UNIT})")
    hist.set_ylabel("Horas")
    hist.text(
        0.98,
        0.95,
        f"media {residuals.mean():.1f}\ndesv. est. {residuals.std():.1f}",
        transform=hist.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    return save(fig, path, dpi)


def plot_error_by_hour(
    predictions: pd.DataFrame, path: Path, dpi: int, peak_hours: list[int]
) -> Path:
    """Error absoluto del campeón por hora del día, con el MAE de los baselines."""
    hours = predictions["fecha_objetivo"].dt.hour
    abs_error = (predictions[CHAMPION] - predictions["real"]).abs()
    groups = [abs_error[hours == h].to_numpy() for h in range(24)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for h in peak_hours:
        ax.axvspan(h - 0.5, h + 0.5, color=COLORS["naive_24h"], alpha=0.08, lw=0)
    ax.boxplot(
        groups,
        positions=range(24),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": COLORS[CHAMPION], "alpha": 0.35},
        medianprops={"color": COLORS[CHAMPION]},
    )
    for name in baselines(predictions)[:1]:
        mae = (predictions[name] - predictions["real"]).abs().groupby(hours).mean()
        ax.plot(
            mae.index, mae, color=COLORS[name], marker="o", ms=3, lw=1, label=f"MAE {LABELS[name]}"
        )
    mae_champion = abs_error.groupby(hours).mean()
    ax.plot(
        mae_champion.index,
        mae_champion,
        color=COLORS[CHAMPION],
        marker="o",
        ms=3,
        lw=1,
        label="MAE campeón",
    )
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día objetivo (sombreado: horas de punta)")
    ax.set_ylabel(f"Error absoluto ({UNIT})")
    ax.set_title("Error del campeón por hora del día (cajas sin atípicos)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, path, dpi)


def plot_mae_by_horizon(by_horizon: pd.DataFrame, path: Path, dpi: int) -> Path:
    """MAE por hora de anticipación de cada modelo."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name in by_horizon.columns:
        ax.plot(
            by_horizon.index,
            by_horizon[name],
            color=COLORS.get(name, "#444444"),
            lw=1.6 if name == CHAMPION else 1.0,
            ls="-" if name == CHAMPION else "--",
            label=LABELS.get(name, name),
        )
    ax.set_xlabel("Horas de anticipación")
    ax.set_ylabel(f"MAE ({UNIT})")
    ax.set_title("Error según la anticipación del pronóstico")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return save(fig, path, dpi)


# -----------------------------------------------------------------------------
# GRADIENTES INTEGRADOS
# -----------------------------------------------------------------------------
def integrated_gradients(
    model: torch.nn.Module, x: np.ndarray, *, output_index: int = 0, steps: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Atribuciones del pronóstico medio del horizonte respecto a cada entrada.

    Usa la regla del punto medio con ``steps`` pasos y la referencia de ceros.

    Args:
        model: Red en modo evaluación que devuelve ``(B, H, salidas)``.
        x: Ventanas ``(B, L, F)`` en la escala del modelo.
        output_index: Salida a explicar (0, o el índice del cuantil 0,5).
        steps: Pasos de la integral numérica.

    Returns:
        Tupla ``(atribuciones (B, L, F), error de completitud relativo por muestra (B,))``.
    """
    model.eval()
    inputs = torch.from_numpy(x).float()
    reference = torch.zeros_like(inputs)
    delta = inputs - reference
    total_grad = torch.zeros_like(inputs)
    for alpha in (np.arange(steps) + 0.5) / steps:
        point = (reference + float(alpha) * delta).requires_grad_(True)
        output = model(point)[..., output_index].mean(dim=1).sum()
        (grad,) = torch.autograd.grad(output, point)
        total_grad += grad
    attributions = (delta * total_grad / steps).numpy()
    with torch.no_grad():
        gap = (
            model(inputs)[..., output_index].mean(dim=1)
            - model(reference)[..., output_index].mean(dim=1)
        ).numpy()
    error = np.abs(attributions.sum(axis=(1, 2)) - gap) / np.maximum(np.abs(gap), 1e-8)
    return attributions, error


def importance_tables(
    attributions: np.ndarray, feature_names: list[str], target_scale: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Importancia media absoluta por variable (escalada y en COP/kWh) y por posición temporal."""
    by_feature = np.abs(attributions).mean(axis=(0, 1)) * attributions.shape[1]
    features = (
        pd.DataFrame(
            {
                "variable": feature_names,
                "importancia": by_feature,
                "importancia_cop_kwh": by_feature * target_scale,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    lookback = attributions.shape[1]
    by_lag = pd.Series(
        np.abs(attributions).sum(axis=2).mean(axis=0) * target_scale,
        index=pd.RangeIndex(lookback, 0, -1, name="horas_antes_del_corte"),
        name="importancia_cop_kwh",
    )
    return features, by_lag


def plot_feature_importance(features: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Barras horizontales con las variables más influyentes."""
    top = features.head(TOP_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(top["variable"], top["importancia_cop_kwh"], color=COLORS[CHAMPION], alpha=0.85)
    ax.set_xlabel(f"|Atribución| media sobre el pronóstico de 24 h ({UNIT})")
    ax.set_title("Variables que más mueven el pronóstico (gradientes integrados)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, path, dpi)


def plot_temporal_importance(by_lag: pd.Series, path: Path, dpi: int) -> Path:
    """Atribución total según cuántas horas antes del corte ocurrió la entrada."""
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(by_lag.index, by_lag.to_numpy(), color=COLORS[CHAMPION], lw=1.2)
    for lag in (24, 48, 72, 96, 168):
        if lag <= by_lag.index.max():
            ax.axvline(lag, color=COLORS["naive_168h"], lw=0.7, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("Horas antes del corte de la ventana")
    ax.set_ylabel(f"|Atribución| ({UNIT})")
    ax.set_title("Qué horas de la ventana usa el modelo")
    fig.tight_layout()
    return save(fig, path, dpi)


def explain_champion(config: dict, root: Path, figures: Path, metrics: Path) -> dict | None:
    """Calcula gradientes integrados del campeón en prueba y exporta tabla y figuras.

    Returns:
        Resumen de la explicación, o ``None`` si el campeón no es una red PyTorch.
    """
    evaluation = config["evaluation"]
    champion = json.loads(
        (root / config["paths"]["models"]["trained"] / CHAMPION_META_FILE).read_text(
            encoding="utf-8"
        )
    )
    if champion["kind"] != "torch":
        logger.warning(
            "El campeón (%s) no es una red: se omiten los gradientes integrados",
            champion["architecture"],
        )
        return None
    model, meta = load_network(Path(champion["path"]))
    test, metadata, scalers = load_test(config, root)
    x = test.x[:, -meta["lookback"] :]
    output_index = meta["quantiles"].index(MEDIAN) if meta["loss"] == "quantile" else 0
    attributions, error = integrated_gradients(
        model, x, output_index=output_index, steps=evaluation["integrated_gradients_steps"]
    )
    target_scale = float(scalers["target"].scale_[0])
    features, by_lag = importance_tables(attributions, metadata["feature_columns"], target_scale)
    features.to_csv(metrics / IMPORTANCE_FILE, index=False, float_format="%.6f")
    dpi = evaluation["dpi"]
    plot_feature_importance(features, figures / "importancia_variables.png", dpi)
    plot_temporal_importance(by_lag, figures / "importancia_temporal.png", dpi)
    summary = {
        "samples": len(x),
        "steps": evaluation["integrated_gradients_steps"],
        "completeness_error_median": float(np.median(error)),
        "top_features": features.head(5)["variable"].tolist(),
    }
    logger.info(
        "Gradientes integrados: error de completitud mediano %.2e; top %s",
        summary["completeness_error_median"],
        summary["top_features"],
    )
    return summary


def run(config: dict, *, root: Path = PROJECT_PATH) -> dict:
    """Genera todas las figuras de evaluación a partir de ``reports/metrics/``.

    Raises:
        FileNotFoundError: Si aún no se ejecutó ``src.models.predict_model``.
    """
    paths, evaluation = config["paths"], config["evaluation"]
    metrics = root / paths["reports"]["metrics"]
    predictions_path = metrics / PREDICTIONS_FILE
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"No existe {predictions_path}; ejecuta primero `python -m src.models.predict_model`."
        )
    predictions = pd.read_parquet(predictions_path)
    by_horizon = pd.read_csv(metrics / HORIZON_FILE, index_col=0)
    figures = root / paths["reports"]["figures"] / evaluation["figures_subdir"]
    dpi = evaluation["dpi"]
    written = [
        plot_forecast(predictions, figures / "serie_real_vs_pronostico.png", dpi),
        plot_residuals(predictions, figures / "residuales.png", dpi),
        plot_error_by_hour(
            predictions, figures / "error_por_hora.png", dpi, evaluation["peak_hours"]
        ),
        plot_mae_by_horizon(by_horizon, figures / "mae_por_horizonte.png", dpi),
    ]
    explanation = explain_champion(config, root, figures, metrics)
    return {
        "figures": [str(p) for p in written],
        "explanation": explanation,
        "directory": str(figures),
    }


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos de las figuras."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        result = run(read_config_mapping(args.config))
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo al generar las figuras: %s", exc)
        log_event(logging.ERROR, "figures_failed", error=str(exc))
        return 1
    logger.info("Figuras en %s", result["directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
