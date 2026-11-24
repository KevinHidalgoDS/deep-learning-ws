# -*- coding: utf-8 -*-  # noqa: UP009
"""Preprocesamiento causal de las series de SIMEM (Sprint 2).

Reconstruye las fuentes de un snapshot inmutable de ``data/raw/simem/`` y las alinea en una
malla horaria común sin usar información del futuro:

- Las series horarias (precio, demanda) se reindexan sobre la malla completa de la ventana del
  snapshot; las horas sin registro quedan explícitas como faltantes.
- Las series diarias (aportes hídricos, generación) se consolidan a una fila por día. Como un dato
  del día D solo se conoce al cierre de ese día, su valor se asigna a las horas a partir de
  D + ``daily_publication_lag_days``.
- Los faltantes se rellenan únicamente hacia adelante (``ffill``) con un límite de pasos y se
  marcan en columnas ``*_imputado``. Nunca se usa ``bfill`` ni interpolación.

Las fuentes configuradas que no estén en el snapshot se registran como omitidas y no se inventan.

Ejemplo:
    Para preprocesar el snapshot más reciente::

        $ python -m src.data.preprocess
        $ python -m src.data.preprocess --snapshot data/raw/simem/2026-11-18/<id>

Proyecto: deep-learning-ws

Programa: preprocess.py

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
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.data.make_dataset import IngestionError, read_snapshot
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
HOURLY_FILE = "simem_hourly.parquet"
DAILY_FILE = "simem_daily.parquet"
REPORT_FILE = "preprocess_report.json"
IMPUTED_SUFFIX = "_imputado"
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


def latest_snapshot(raw_dir: Path) -> Path:
    """Devuelve el snapshot de SIMEM publicado más recientemente.

    Los directorios siguen el patrón ``simem/AAAA-MM-DD/HHMMSSffffffZ_<id>``, por lo que el
    orden lexicográfico coincide con el orden de captura.

    Args:
        raw_dir: Directorio ``data/raw/``.

    Returns:
        Ruta del directorio del snapshot más reciente que tiene ``manifest.json``.

    Raises:
        FileNotFoundError: Si no existe ningún snapshot publicado.
    """
    manifests = sorted(raw_dir.glob("simem/*/*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No hay snapshots de SIMEM en {raw_dir / 'simem'}; "
            "ejecuta primero `python -m src.data.make_dataset`."
        )
    return manifests[-1].parent


def value_column(name: str, target_column: str) -> str:
    """Nombre de la columna de valor que ``read_snapshot`` asigna a cada fuente."""
    return target_column if name == "precio_bolsa" else name


def load_sources(
    snapshot: Path, names: list[str], target_column: str
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, str]]:
    """Reconstruye las fuentes configuradas presentes en un snapshot.

    Args:
        snapshot: Directorio del snapshot publicado.
        names: Fuentes configuradas en ``simem.datasets``.
        target_column: Nombre de la columna objetivo (precio).

    Returns:
        Tupla ``(fuentes, omitidas, ventana)`` con los DataFrames reconstruidos, las fuentes
        configuradas que no están en el snapshot y la ventana ``{"start", "end"}`` del manifiesto.

    Raises:
        IngestionError: Si el snapshot no contiene el precio de bolsa.
    """
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    available = manifest.get("datasets", {})
    sources, omitted = {}, []
    for name in names:
        if name not in available:
            logger.warning("Fuente '%s' configurada pero ausente en el snapshot; se omite", name)
            omitted.append(name)
            continue
        sources[name] = read_snapshot(snapshot, name)
    if "precio_bolsa" not in sources:
        raise IngestionError(f"El snapshot {snapshot} no contiene precio_bolsa")
    if sources["precio_bolsa"].columns[-1] != target_column:
        raise IngestionError(f"El precio no trae la columna objetivo '{target_column}'")
    return sources, omitted, manifest["window"]


def hourly_grid(start: str, end: str, timezone: str, frequency: str = "1h") -> pd.DatetimeIndex:
    """Malla horaria completa de la ventana ``[start, end]`` (fechas inclusivas, hora local)."""
    first = pd.Timestamp(start, tz=timezone)
    stop = pd.Timestamp(end, tz=timezone) + pd.DateOffset(days=1)
    return pd.date_range(first, stop, freq=frequency, inclusive="left", name="fecha")


def to_grid(frame: pd.DataFrame, column: str, grid: pd.DatetimeIndex) -> pd.Series:
    """Reindexa una serie de una sola clave temporal sobre una malla dada.

    Raises:
        ValueError: Si la serie tiene marcas temporales repetidas.
    """
    series = frame.set_index("fecha")[column]
    if not series.index.is_unique:
        raise ValueError(f"'{column}' tiene marcas temporales duplicadas")
    if outside := len(series.index.difference(grid)):
        logger.warning("%s: %d registros fuera de la malla se descartan", column, outside)
    return series.reindex(grid)


def _slug(text: object) -> str:
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def daily_table(frame: pd.DataFrame, name: str, spec: dict) -> pd.DataFrame:
    """Consolida una fuente diaria a una fila por día.

    - Con ``aggregate`` y ``group_by`` (generación por tipo): una columna por grupo,
      p. ej. ``generacion_real_hidraulica``.
    - Con dimensiones sin agregar (aportes por serie hidrológica): suma de todas las series; un
      día con alguna serie faltante queda como faltante en lugar de subestimar el total.
    - Sin dimensiones: la serie tal cual.

    Args:
        frame: Fuente reconstruida con ``read_snapshot``.
        name: Nombre de la fuente (también es su columna de valor).
        spec: Especificación de la fuente en ``simem.datasets``.

    Returns:
        DataFrame indexado por la fecha local (medianoche) con una o más columnas numéricas.
    """
    group_by = spec.get("group_by") or []
    dimensions = [c for c in frame.columns if c not in ("fecha", name)]
    if spec.get("aggregate") and group_by:
        wide = frame.set_index(["fecha", *group_by])[name].unstack(group_by)
        wide.columns = [
            f"{name}_{_slug('_'.join(map(str, c)) if isinstance(c, tuple) else c)}"
            for c in wide.columns
        ]
        return wide.sort_index()
    if dimensions:
        n_series = len(frame[dimensions].drop_duplicates())
        grouped = frame.groupby("fecha")[name]
        total = grouped.sum().where(grouped.count().eq(n_series))
        if incomplete := int(total.isna().sum()):
            logger.warning(
                "%s: %d días sin las %d series completas quedan como faltantes",
                name,
                incomplete,
                n_series,
            )
        return total.to_frame(name).sort_index()
    return frame.set_index("fecha")[[name]].sort_index()


def causal_impute(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Rellena faltantes solo con el último valor observado y marca lo imputado.

    Cada faltante toma el último valor anterior disponible, como máximo ``limit`` pasos
    consecutivos. Los faltantes al inicio de la serie o en huecos más largos que ``limit``
    permanecen como ``NaN``: completarlos exigiría mirar al futuro.

    Args:
        frame: Series alineadas en orden cronológico.
        limit: Máximo de pasos consecutivos a rellenar.

    Returns:
        Copia con los valores imputados y una columna ``<col>_imputado`` (0/1) por columna.
    """
    if not frame.index.is_monotonic_increasing:
        raise ValueError("La imputación causal requiere un índice cronológico creciente")
    filled = frame.ffill(limit=limit)
    flags = (frame.isna() & filled.notna()).astype("int8").add_suffix(IMPUTED_SUFFIX)
    return pd.concat([filled, flags], axis=1)


def align_daily_causal(daily: pd.DataFrame, grid: pd.DatetimeIndex, lag_days: int) -> pd.DataFrame:
    """Asigna a cada hora el último dato diario ya publicado.

    La hora ``h`` del día D recibe el valor del día ``D - lag_days``.

    Args:
        daily: Tabla diaria indexada por medianoche local.
        grid: Malla horaria destino.
        lag_days: Días de rezago de publicación (al menos 1).

    Returns:
        Tabla con el mismo índice que ``grid``.

    Raises:
        ValueError: Si ``lag_days < 1``, porque usaría el dato del día en curso.
    """
    if lag_days < 1:
        raise ValueError("daily_publication_lag_days debe ser >= 1 para no usar el día en curso")
    # Se desplaza en hora local sin zona y con unidad explícita (días de calendario).
    local_days = daily.index.normalize()
    published = pd.DatetimeIndex(
        local_days.tz_localize(None).to_numpy() + np.timedelta64(lag_days, "D")
    ).tz_localize(local_days.tz)
    return daily.set_axis(published).reindex(grid.normalize()).set_axis(grid)


def build_tables(
    sources: dict[str, pd.DataFrame],
    specs: dict[str, dict],
    window: dict[str, str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Construye las tablas intermedias horaria (para modelar) y diaria (descriptiva).

    Args:
        sources: Fuentes reconstruidas por nombre; debe incluir ``precio_bolsa``.
        specs: Sección ``simem.datasets`` de la configuración.
        window: Ventana ``{"start", "end"}`` del snapshot.
        config: Configuración completa (usa ``data`` y ``preprocess``).

    Returns:
        Tupla ``(horaria, diaria, reporte)``. La horaria contiene las series horarias imputadas
        y las diarias alineadas causalmente; la diaria resume cada fuente en su propia fecha
        (sin desplazar) y no debe usarse como predictor.
    """
    settings, params = config["data"], config["preprocess"]
    target = settings["target_column"]
    grid = hourly_grid(window["start"], window["end"], settings["timezone"], settings["frequency"])
    days = pd.date_range(grid[0], grid[-1].normalize(), freq="D", name="fecha")

    hourly_names = [n for n in sources if specs[n]["frequency"] == "1h"]
    daily_names = [n for n in sources if specs[n]["frequency"] == "1D"]

    # 1. Series horarias sobre la malla completa
    raw_hourly = pd.DataFrame(
        {
            value_column(n, target): to_grid(sources[n], value_column(n, target), grid)
            for n in hourly_names
        },
        index=grid,
    )
    hourly = causal_impute(raw_hourly, params["max_gap_hours"])

    # 2. Series diarias: un valor por día, imputación en días y alineación causal a horas
    raw_daily = pd.DataFrame(index=days)
    if daily_names:
        raw_daily = pd.concat(
            [daily_table(sources[n], n, specs[n]) for n in daily_names], axis=1
        ).reindex(days)
        daily = causal_impute(raw_daily, params["max_gap_days"])
        hourly = hourly.join(align_daily_causal(daily, grid, params["daily_publication_lag_days"]))

    # 3. Vista diaria descriptiva: solo días con las 24 horas observadas
    complete = raw_hourly.resample("D").count().eq(HOURS_PER_DAY)
    daily_view = pd.concat(
        [
            raw_hourly[[target]]
            .resample("D")
            .agg(["mean", "min", "max"])
            .set_axis([f"{target}_media", f"{target}_min", f"{target}_max"], axis=1)
            .where(complete[target], axis=0),
            raw_hourly.drop(columns=target)
            .resample("D")
            .sum()
            .add_suffix("_diaria")
            .where(complete.drop(columns=target).to_numpy()),
            raw_daily,
        ],
        axis=1,
    )
    daily_view.index.name = "fecha"

    imputed = {
        c.removesuffix(IMPUTED_SUFFIX): int(hourly[c].sum())
        for c in hourly.columns
        if c.endswith(IMPUTED_SUFFIX)
    }
    report = {
        "window": window,
        "rows": len(hourly),
        "imputed": imputed,
        "missing_after_imputation": {c: int(n) for c, n in hourly.isna().sum().items() if n},
    }
    return hourly, daily_view, report


def run(config: dict, *, snapshot: Path | None = None, root: Path = PROJECT_PATH) -> Path:
    """Ejecuta el preprocesamiento y guarda las tablas en ``data/interim/``.

    Args:
        config: Configuración completa.
        snapshot: Snapshot a procesar; por defecto, el más reciente.
        root: Raíz del proyecto.

    Returns:
        Ruta del parquet horario generado.
    """
    paths = config["paths"]["data"]
    snapshot = snapshot or latest_snapshot(root / paths["raw"])
    specs = config["simem"]["datasets"]
    logger.info("Preprocesando snapshot %s", snapshot)

    sources, omitted, window = load_sources(snapshot, list(specs), config["data"]["target_column"])
    hourly, daily, report = build_tables(sources, specs, window, config)

    interim = root / paths["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    hourly_path = interim / HOURLY_FILE
    hourly.reset_index().to_parquet(hourly_path, index=False)
    daily.reset_index().to_parquet(interim / DAILY_FILE, index=False)
    report.update(snapshot=str(snapshot), omitted_sources=omitted, columns=list(hourly.columns))
    (interim / REPORT_FILE).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    logger.info("Tabla horaria guardada en %s (%d filas)", hourly_path, len(hourly))
    log_event(logging.INFO, "interim_published", path=hourly_path, **report)
    return hourly_path


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos del preprocesamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--snapshot", type=Path, help="Snapshot de SIMEM (por defecto, el último)")
    args = parser.parse_args(argv)
    try:
        run(read_config_mapping(args.config), snapshot=args.snapshot)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Fallo en el preprocesamiento: %s", exc)
        log_event(logging.ERROR, "preprocess_failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
