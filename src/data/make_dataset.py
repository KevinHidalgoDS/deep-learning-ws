# -*- coding: utf-8 -*-  # noqa: UP009
"""Módulo de ingestión y procesamiento reproducible de datos de SIMEM.

Este módulo implementa el pipeline de descarga, validación de contratos de esquema,
normalización de series temporales, resolución de calendarios oficiales de liquidación
y almacenamiento reproducible de snapshots con manifiestos criptográficos (SHA-256).

Ejemplo:
    Para ejecutar la ingesta desde la línea de comandos::

        $ python -m src.data.make_dataset --start 2024-04-01 --end 2024-04-30

Proyecto: deep-learning-ws

Programa: make_dataset.py

Soporte: kfhidalgoh@unal.edu.co

version: 1.0.0

lenguaje: Python 3.14.5

CD: 20261118

LUD: 20261118

Comentarios:
    - 2026-11-18 Kevin Hidalgo -> creación.
"""

__authors__ = ["Kevin Hidalgo"]
__contact__ = "kfhidalgoh@unal.edu.co"
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Kevin Hidalgo"]
__email__ = "kfhidalgoh@unal.edu.co"
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-18"

import argparse
import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import DEFAULT_CONFIG_PATH, PROJECT_PATH, read_config_mapping
from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGING DEL MÓDULO
# -----------------------------------------------------------------------------
MAX_CHUNK_DAYS = 31

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


class IngestionError(ValueError):
    """Error de configuración, contrato de datos o calidad de la fuente en SIMEM.

    Esta excepción personalizada se utiliza para diagnosticar problemas de esquema,
    inconsistencias temporales, ausencia de versiones oficiales de liquidación o fallas
    en la integridad criptográfica de los datos ingeridos.

    Args:
        message: Descripción detallada del error o inconsistencia detectada.

    Examples:
        >>> raise IngestionError("Columnas de origen faltantes: ['Valor']")
        Traceback (most recent call last):
            ...
        src.data.make_dataset.IngestionError: Columnas de origen faltantes: ['Valor']
    """


@dataclass(frozen=True)
class Dataset:
    """Especificación y reglas de transformación de un conjunto de datos de SIMEM.

    Define los metadatos necesarios para extraer, filtrar, normalizar y validar
    un dataset específico de la plataforma SIMEM.

    Attributes:
        dataset_id: Código identificador alfanumérico único de seis caracteres en SIMEM.
        timestamp: Nombre de la columna que contiene la fecha y hora original.
        value: Nombre de la columna que contiene la métrica numérica principal.
        frequency: Frecuencia de la serie temporal (ej. '1h' para horaria, '1D' para diaria).
        filters: Mapeo de columnas y valores permitidos para filtrar los registros.
        dimensions: Lista de columnas categóricas que componen la clave dimensional.
        group_by: Columnas para la agregación espacial/dimensional cuando aggregate=True.
        aggregate: Indica si se debe consolidar la serie sumando sobre el grupo dimensional.
        revision: Nombre de la columna que contiene la versión de liquidación (ej. 'Version').
        revision_calendar_id: Identificador de seis caracteres del dataset de calendario oficial.
        revision_as_of: Fecha de corte 'AAAA-MM-DD' para reconstrucción histórica as-of.
        revision_fallback: Si permite usar la versión más reciente disponible ante ausencia oficial.
    """

    dataset_id: str
    timestamp: str
    value: str
    frequency: str
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    aggregate: bool = False
    revision: str | None = None
    revision_calendar_id: str | None = None
    revision_as_of: str | None = None
    revision_fallback: bool = False


def date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """Divide un rango inclusivo de fechas en ventanas consecutivas sin solapamiento.

    Args:
        start: Fecha inicial inclusiva del período.
        end: Fecha final inclusiva del período.
        days: Tamaño máximo en días para cada fragmento (entre 1 y MAX_CHUNK_DAYS).

    Yields:
        Tuplas (fecha_inicio, fecha_fin) para cada ventana de consulta.

    Raises:
        IngestionError: Si start > end o si days no está en el rango [1, MAX_CHUNK_DAYS].

    Examples:
        >>> list(date_chunks(date(2024, 4, 1), date(2024, 4, 3), 2))
        [(date(2024, 4, 1), date(2024, 4, 2)), (date(2024, 4, 3), date(2024, 4, 3))]
    """
    # 1. Validación de precondiciones del rango y tamaño de partición
    logger.debug(
        "Calculando fragmentos de fechas desde %s hasta %s con paso de %d días",
        start,
        end,
        days,
    )
    if start > end or not 1 <= days <= MAX_CHUNK_DAYS:
        logger.error(
            "Rango invertido (start=%s, end=%s) o days=%d fuera de [1, %d]",
            start,
            end,
            days,
            MAX_CHUNK_DAYS,
        )
        raise IngestionError(f"Rango invertido o chunk_days fuera de [1, {MAX_CHUNK_DAYS}]")

    # 2. Generación iterativa de ventanas temporales consecutivas
    while start <= end:
        last = start + timedelta(days=min(days - 1, (end - start).days))
        yield start, last
        if last == end:
            break
        start = last + timedelta(days=1)


def records_from_bytes(content: bytes, suffix: str) -> pd.DataFrame:
    """Lee y decodifica un contenido en bytes a DataFrame (soporta CSV y JSON de PublicData).

    Args:
        content: Contenido binario del archivo o respuesta del servidor.
        suffix: Extensión o sufijo del archivo (ej. '.csv' o '.json').

    Returns:
        DataFrame de pandas con los registros procesados.

    Raises:
        IngestionError: Si el contenido no cumple el formato o contrato de SIMEM.

    Examples:
        >>> df = records_from_bytes(b'{"success": true, "result": {"records": []}}', ".json")
        >>> isinstance(df, pd.DataFrame)
        True
    """
    # 1. Procesamiento de archivos en formato CSV
    if suffix == ".csv":
        logger.debug("Decodificando fragmento en formato CSV (%d bytes)", len(content))
        try:
            return pd.read_csv(
                BytesIO(content), dtype="string", keep_default_na=False, na_values=[""]
            )
        except Exception as exc:
            logger.error("Error al parsear archivo CSV: %s", exc)
            raise IngestionError(f"Error al parsear contenido CSV: {exc}") from exc

    # 2. Deserialización de respuesta JSON de la API
    logger.debug("Decodificando fragmento en formato JSON (%d bytes)", len(content))
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Error de decodificación JSON: %s", exc)
        raise IngestionError(f"Respuesta no es un JSON válido: {exc}") from exc

    # 3. Validación del contrato de éxito de la API de SIMEM
    if not isinstance(payload, dict) or payload.get("success") is not True:
        logger.error("Contrato de SIMEM incumplido: payload no exitoso o estructura no válida")
        raise IngestionError("SIMEM no devolvió success=true")

    # 4. Extracción de registros del resultado
    result = payload.get("result")
    if result is None and payload.get("hasData") is False:
        logger.debug("La respuesta de SIMEM indica que no hay datos disponibles (hasData=False)")
        return pd.DataFrame()

    records = result.get("records") if isinstance(result, dict) else None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        logger.error("Estructura inválida en result.records: se esperaba lista de diccionarios")
        raise IngestionError("SIMEM: se esperaba una lista de objetos en result.records")

    return pd.DataFrame.from_records(records)


def _normalize_chunk(raw: pd.DataFrame, spec: Dataset, timezone: str) -> pd.DataFrame:
    """Normaliza un fragmento crudo validando columnas requeridas, tipos y malla temporal.

    Args:
        raw: DataFrame con los registros crudos de la fuente.
        spec: Especificación de configuración del dataset.
        timezone: Zona horaria destino (ej. 'America/Bogota').

    Returns:
        DataFrame normalizado con columnas tipadas ('fecha', 'valor', dimensiones y versión).

    Raises:
        IngestionError: Si faltan columnas requeridas, si hay timestamps nulos o fuera de
            malla, si hay valores numéricos infinitos o si las dimensiones contienen nulos.
        ValueError: Si falla la conversión de tipos en fecha o valor numérico.

    Examples:
        >>> # Uso interno durante la preparación de fragmentos
    """
    logger.debug("Normalizando fragmento de dataset '%s' (%d filas)", spec.dataset_id, len(raw))

    # 1. Validación de existencia de columnas requeridas por la especificación
    required = {spec.timestamp, spec.value, *spec.filters, *spec.dimensions}
    if spec.revision:
        required.add(spec.revision)
    if missing := required - set(raw.columns):
        logger.error("Columnas faltantes en dataset '%s': %s", spec.dataset_id, sorted(missing))
        raise IngestionError(f"Columnas de origen faltantes: {sorted(missing)}")

    # 2. Aplicación de filtros categóricos sobre los datos
    for column, choices in spec.filters.items():
        allowed = [choices] if isinstance(choices, str) else choices
        raw = raw.loc[raw[column].isin(allowed)]

    # 3. Conversión, localización y validación de marcas temporales
    try:
        timestamps = pd.to_datetime(raw[spec.timestamp], errors="raise", format="ISO8601")
    except Exception as exc:
        logger.error("Error al convertir fechas en columna '%s': %s", spec.timestamp, exc)
        raise

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone)
    else:
        timestamps = timestamps.dt.tz_convert(timezone)

    if timestamps.isna().any() or not timestamps.eq(timestamps.dt.floor(spec.frequency)).all():
        logger.error(
            "Se detectaron fechas nulas o fuera de la malla de frecuencia '%s'", spec.frequency
        )
        raise IngestionError(f"Fechas nulas o fuera de la malla {spec.frequency}")

    # 4. Conversión y verificación de valores numéricos
    try:
        values = pd.to_numeric(raw[spec.value], errors="raise").astype("float64")
    except Exception as exc:
        logger.error("Error al convertir columna de valor '%s': %s", spec.value, exc)
        raise

    if np.isinf(values).any():
        logger.error("Valores infinitos detectados en la columna numérica '%s'", spec.value)
        raise IngestionError("Valores infinitos en la columna numérica")

    # 5. Estructuración del DataFrame normalizado y validación de dimensiones
    clean = raw[spec.dimensions].astype("string").assign(fecha=timestamps, valor=values)
    if spec.revision:
        clean["revision"] = raw[spec.revision].astype("string").str.strip()

    if clean[spec.dimensions].isna().any().any():
        logger.error("Se detectaron valores nulos en columnas dimensionales: %s", spec.dimensions)
        raise IngestionError("Dimensiones nulas en la clave temporal")

    return clean


def _official_revisions(
    calendar: pd.DataFrame, start: date, end: date, as_of: str | None
) -> dict[str, str]:
    """Determina la versión oficial de liquidación para cada mes en un rango de fechas.

    Args:
        calendar: DataFrame con los registros del calendario oficial de liquidación.
        start: Fecha inicial del rango de consulta.
        end: Fecha final del rango de consulta.
        as_of: Fecha de corte 'AAAA-MM-DD' para simulación histórica (o None para máxima vigente).

    Returns:
        Diccionario asociando cada período mensual 'AAAA-MM' con su versión oficial (ej. 'TX5').

    Raises:
        IngestionError: Si faltan columnas requeridas, si existen fechas o versiones nulas,
            o si no existe una única revisión oficial por período mensual.

    Examples:
        >>> # Cálculo de revisiones oficiales por mes a partir de calendario
    """
    logger.debug(
        "Calculando revisiones oficiales para el período %s a %s (corte: %s)", start, end, as_of
    )

    # 1. Validación de columnas obligatorias del calendario
    required = {"Version", "FechaInicio", "FechaFin", "FechaPublicacion", "EsMaximaVersion"}
    if missing := required - set(calendar.columns):
        logger.error("Columnas faltantes en calendario de liquidación: %s", sorted(missing))
        raise IngestionError(f"Columnas faltantes en calendario de liquidación: {sorted(missing)}")

    # 2. Limpieza, tipado y verificación de valores no nulos
    data = calendar[list(required)].drop_duplicates().copy()
    dates = ["FechaInicio", "FechaFin", "FechaPublicacion"]
    try:
        data[dates] = data[dates].apply(pd.to_datetime, format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        logger.error("Error al formatear fechas del calendario de liquidación: %s", exc)
        raise IngestionError(f"Error al formatear fechas del calendario: {exc}") from exc

    data["Version"] = data["Version"].astype("string").str.strip()
    if data[dates].isna().any().any() or data["Version"].fillna("").eq("").any():
        logger.error("El calendario contiene fechas o versiones de liquidación nulas o vacías")
        raise IngestionError("El calendario contiene fechas o versiones de liquidación nulas")

    # 3. Selección de versión oficial para cada período mensual
    periods = pd.period_range(start, end, freq="M")
    revisions = {}
    for period in periods:
        rows = data.loc[
            data["FechaInicio"].eq(period.start_time)
            & data["FechaFin"].eq(period.end_time.normalize())
        ]
        if as_of is None:
            eligible = rows.loc[pd.to_numeric(rows["EsMaximaVersion"], errors="raise").eq(1)]
        else:
            as_of_ts = pd.Timestamp(date.fromisoformat(as_of))
            eligible = rows.loc[rows["FechaPublicacion"].le(as_of_ts)]
            eligible = eligible.loc[
                eligible["FechaPublicacion"].eq(eligible["FechaPublicacion"].max())
            ]

        if len(eligible) != 1:
            logger.error(
                "Inconsistencia en revisión oficial para %s (encontradas: %d)",
                period,
                len(eligible),
            )
            raise IngestionError(
                f"Se esperaba una revisión oficial única para {period}"
                f" (corte: {as_of or 'vigente'}); encontradas: {len(eligible)}"
            )
        revisions[str(period)] = eligible["Version"].iloc[0]

    logger.debug("Revisiones oficiales determinadas: %s", revisions)
    return revisions


def _fallback_revisions(calendar: pd.DataFrame, as_of: str | None) -> dict[str, list[list[str]]]:
    """Construye un mapa de versiones alternativas (fallback) ordenadas cronológicamente por mes.

    Args:
        calendar: DataFrame con los registros del calendario de liquidaciones.
        as_of: Fecha de corte opcional 'AAAA-MM-DD' para filtrar publicaciones disponibles.

    Returns:
        Diccionario que mapea cada período 'AAAA-MM' con listas de versiones agrupadas por lote,
        ordenadas de más reciente a más antigua.

    Examples:
        >>> # Generación de jerarquía de fallback para versiones
    """
    logger.debug("Construyendo árbol de versiones de fallback (corte: %s)", as_of)

    # 1. Extracción de fechas de publicación y asignación de período mensual
    published = pd.to_datetime(calendar["FechaPublicacion"])
    rows = calendar.assign(published=published, period=calendar["FechaInicio"].str[:7])

    # 2. Filtrado por fecha de corte as-of si está presente
    if as_of is not None:
        rows = rows.loc[rows["published"].le(pd.Timestamp(as_of))]

    # 3. Agrupación por período y ordenamiento inverso por fecha de publicación
    return {
        period: [
            sorted(batch["Version"].str.strip().unique())
            for _, batch in month.groupby("published", sort=True)
        ][::-1]
        for period, month in rows.groupby("period")
    }


def _resolve_available(
    data: pd.DataFrame, preferred: dict[str, str], fallback: dict[str, list[list[str]]]
) -> dict[str, str]:
    """Resuelve la versión a usar seleccionando fallback si la preferida no cubre todas las horas.

    Args:
        data: DataFrame normalizado que contiene columnas 'fecha' y 'revision'.
        preferred: Diccionario con las versiones preferidas por período 'AAAA-MM'.
        fallback: Diccionario con los lotes de versiones de respaldo por período.

    Returns:
        Diccionario con las versiones definitivas que poseen cobertura completa.

    Raises:
        IngestionError: Si existen versiones alternativas ambiguas en el mismo lote de fallback.

    Examples:
        >>> # Selección de versión fallback ante ausencia de la versión oficial
    """
    logger.debug("Resolviendo cobertura de versiones disponibles versus preferidas")

    # 1. Identificación de combinaciones únicas de fecha y revisión
    distinct = data[["fecha", "revision"]].drop_duplicates()
    distinct = distinct.assign(period=distinct["fecha"].dt.strftime("%Y-%m"))
    selected = preferred.copy()

    # 2. Verificación de cobertura completa por período y resolución vía fallback
    for period, month in distinct.groupby("period"):
        counts = month.groupby("revision")["fecha"].nunique()
        total_unique_dates = month["fecha"].nunique()
        complete = set(counts.index[counts.eq(total_unique_dates)])

        # Si la versión preferida tiene cobertura total, conservarla
        if preferred.get(period) in complete:
            continue

        logger.warning(
            "Versión preferida '%s' incompleta para el período %s; evaluando fallback",
            preferred.get(period),
            period,
        )

        # Evaluar lotes de fallback en orden de preferencia
        for batch in fallback.get(period, []):
            eligible = complete.intersection(batch)
            if len(eligible) > 1:
                logger.error(
                    "Revisiones alternativas ambiguas para %s: %s", period, sorted(eligible)
                )
                raise IngestionError(
                    f"Revisiones alternativas ambiguas para {period}: {sorted(eligible)}"
                )
            if eligible:
                selected[period] = eligible.pop()
                logger.info(
                    "Seleccionada versión fallback '%s' para el período %s",
                    selected[period],
                    period,
                )
                break

    return selected


def _select_revisions(
    data: pd.DataFrame,
    revisions: dict[str, str] | None,
    fallback: dict[str, list[list[str]]] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Filtra y valida que cada marca temporal contenga exactamente una versión de liquidación.

    Args:
        data: DataFrame con las observaciones y columna 'revision'.
        revisions: Diccionario con la versión requerida por período mensual 'AAAA-MM'.
        fallback: Diccionario opcional de versiones de respaldo para fallback.

    Returns:
        Tupla con el DataFrame filtrado a una única versión y el mapeo de revisiones aplicado.

    Raises:
        IngestionError: Si hay versiones nulas o vacías, si la versión requerida está ausente
            en horas con datos, o si existen múltiples versiones simultáneas para una misma fecha.

    Examples:
        >>> # Filtrado y control de consistencia de versiones de liquidación
    """
    logger.debug("Validando y filtrando revisiones de liquidación en el dataset")

    # 1. Validación de versiones nulas o vacías
    invalid = data["revision"].str.strip().fillna("").eq("")
    if invalid.any():
        hours = data.loc[invalid, "fecha"].drop_duplicates()
        logger.error("Versiones de liquidación nulas o vacías detectadas en %d horas", len(hours))
        raise IngestionError(
            f"Versiones de liquidación nulas o vacías en {len(hours)} horas; "
            f"ejemplos: {hours.head(3).astype(str).tolist()}"
        )

    # 2. Filtrado por revisión oficial esperada y validación de cobertura
    if revisions is not None:
        if fallback:
            revisions = _resolve_available(data, revisions, fallback)
        hours = pd.DatetimeIndex(data["fecha"].unique())
        expected = pd.Series(hours.strftime("%Y-%m").map(revisions), index=hours)
        selected = data.loc[data["revision"].eq(data["fecha"].map(expected))]
        missing = hours.difference(selected["fecha"])
        if not missing.empty:
            examples = data.loc[data["fecha"].isin(missing[:3]), ["fecha", "revision"]]
            examples = examples.drop_duplicates().assign(
                fecha=lambda rows: rows["fecha"].astype(str)
            )
            logger.error(
                "Revisión oficial ausente en %d horas con datos; versiones esperadas: %s",
                len(missing),
                revisions,
            )
            raise IngestionError(
                f"Revisión oficial ausente en {len(missing)} horas con datos; "
                f"versiones esperadas: {revisions}; "
                f"ejemplos disponibles: {examples.to_dict('records')}"
            )
        data = selected

    # 3. Control de multiplicidad: no deben coexistir múltiples versiones para una misma fecha
    counts = data.groupby("fecha")["revision"].nunique()
    multiple = counts.loc[counts.gt(1)]
    if not multiple.empty:
        examples = data.loc[data["fecha"].isin(multiple.index[:3]), ["fecha", "revision"]]
        examples = examples.drop_duplicates().assign(fecha=lambda rows: rows["fecha"].astype(str))
        logger.error("Versiones de liquidación múltiples detectadas en %d horas", len(multiple))
        raise IngestionError(
            f"Versiones de liquidación múltiples en {len(multiple)} horas; "
            f"ejemplos: {examples.to_dict('records')}"
        )

    return data, revisions or {}


def normalize_dataset(  # noqa: PLR0913
    chunks: list[pd.DataFrame],
    spec: Dataset,
    start: date,
    end: date,
    timezone: str,
    *,
    timestamp_column: str = "fecha",
    value_column: str = "precio_bolsa",
    revisions: dict[str, str] | None = None,
    fallback: dict[str, list[list[str]]] | None = None,
) -> pd.DataFrame:
    """Valida las claves naturales antes de construir series a su granularidad nativa.

    Args:
        chunks: Lista de DataFrames correspondientes a los fragmentos a ensamblar.
        spec: Especificación de configuración del dataset.
        start: Fecha inicial inclusiva de la ventana temporal.
        end: Fecha final inclusiva de la ventana temporal.
        timezone: Zona horaria para localización/conversión.
        timestamp_column: Nombre asignado a la columna temporal en el resultado.
        value_column: Nombre asignado a la columna de métrica en el resultado.
        revisions: Diccionario opcional de revisiones oficiales requeridas por período.
        fallback: Diccionario opcional de versiones alternativas de respaldo.

    Returns:
        DataFrame estructurado y ordenado por su clave natural, con metadatos de revisión.

    Raises:
        IngestionError: Si no hay fragmentos con datos, si no hay valores en la ventana,
            si falta el calendario de revisiones requerido, si hay registros duplicados en
            la clave natural o si todos los valores agregados son nulos.

    Examples:
        >>> # Normalización de dataset con control de duplicados y filtros temporales
    """
    logger.debug(
        "Iniciando normalización de dataset '%s' con %d fragmentos", spec.dataset_id, len(chunks)
    )

    # 1. Normalización de fragmentos individuales no vacíos
    blocks = [_normalize_chunk(chunk, spec, timezone) for chunk in chunks if len(chunk.index)]
    if not blocks:
        logger.error("Dataset '%s' sin registros en los fragmentos suministrados", spec.dataset_id)
        raise IngestionError("Fuente sin registros")

    # 2. Concatenación y filtrado por ventana temporal solicitada
    data = pd.concat(blocks, ignore_index=True)
    data = data.loc[data["fecha"].dt.date.between(start, end)]
    if data.empty or not data["valor"].notna().any():
        logger.error(
            "Sin valores válidos en la ventana temporal %s a %s para dataset '%s'",
            start,
            end,
            spec.dataset_id,
        )
        raise IngestionError("Sin valores en la ventana para los filtros seleccionados")

    # 3. Tratamiento y selección de revisiones de liquidación si corresponde
    selected = {}
    if spec.revision:
        if spec.revision_calendar_id and revisions is None:
            logger.error(
                "Falta el calendario oficial de revisiones para dataset '%s'", spec.dataset_id
            )
            raise IngestionError("Falta el calendario oficial de revisiones de liquidación")
        data, selected = _select_revisions(data, revisions, fallback)

    # 4. Verificación de duplicados en la clave temporal natural
    natural_key = ["fecha", *spec.dimensions]
    if data.duplicated(natural_key).any():
        logger.error("Registros duplicados detectados en la clave temporal: %s", natural_key)
        raise IngestionError(f"Duplicados en la clave temporal {natural_key}")

    # 5. Agregación dimensional opcional
    key = natural_key
    if spec.aggregate:
        key = ["fecha", *spec.group_by]
        logger.debug("Agregando métricas agrupadas por: %s", key)
        grouped = data.groupby(key, sort=False, dropna=False)["valor"]
        totals = grouped.sum().where(grouped.count().eq(grouped.size()))
        data = totals.reset_index()

    if not data["valor"].notna().any():
        logger.error("Todos los valores resultantes de la agregación son faltantes")
        raise IngestionError("Todos los valores agregados son faltantes")

    # 6. Ordenamiento, renombre de columnas y almacenamiento de metadatos en attrs
    output = (
        data.sort_values(key)[[*key, "valor"]]
        .rename(columns={"fecha": timestamp_column, "valor": value_column})
        .reset_index(drop=True)
    )
    output.attrs["settlement_revisions"] = selected
    logger.debug(
        "Normalización completada para '%s': %d filas generadas", spec.dataset_id, len(output)
    )
    return output


def _session(max_retries: int) -> requests.Session:
    """Configura y retorna una sesión HTTP con reintentos exponenciales para peticiones GET.

    Args:
        max_retries: Cantidad máxima de reintentos ante errores de conexión o HTTP 429/5xx.

    Returns:
        Instancia de `requests.Session` montada con el adaptador de reintentos.

    Examples:
        >>> s = _session(3)
        >>> isinstance(s, requests.Session)
        True
    """
    logger.debug("Inicializando sesión HTTP con max_retries=%d", max_retries)
    # 1. Configuración de política de reintentos de urllib3
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
    )
    # 2. Montaje de adaptador para transporte seguro HTTPS
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session


def _local_chunk(source_dir: Path, name: str) -> tuple[str, bytes, dict[str, str]]:
    """Localiza y lee el contenido binario de un archivo de exportación local (.csv o .json).

    Args:
        source_dir: Directorio local donde residen las exportaciones de SIMEM.
        name: Nombre base del archivo a buscar.

    Returns:
        Tupla con (nombre_archivo, contenido_bytes, diccionario_origen).

    Raises:
        IngestionError: Si no se encuentra exactamente un archivo con extensión .csv o .json.

    Examples:
        >>> # Lectura de archivo local durante pruebas o ingesta offline
    """
    logger.debug("Buscando exportación local para '%s' en '%s'", name, source_dir)
    # 1. Detección de candidatos de archivos compatibles
    candidates = [
        path for suffix in (".csv", ".json") if (path := source_dir / f"{name}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        logger.error(
            "Se requiere exactamente un archivo %s.csv o %s.json en %s (encontrados: %d)",
            name,
            name,
            source_dir,
            len(candidates),
        )
        raise IngestionError(f"Se requiere exactamente un archivo {name}.csv o {name}.json")

    # 2. Lectura del contenido binario y resolución de metadatos de ruta
    path = candidates[0]
    try:
        content = path.read_bytes()
    except OSError as exc:
        logger.error("Error al leer archivo local '%s': %s", path, exc)
        raise IngestionError(f"Error al leer archivo {path}: {exc}") from exc

    return path.name, content, {"source": str(path.resolve())}


def _api_chunks(
    session: requests.Session,
    api: dict,
    dataset_id: str,
    name: str,
    windows: Iterable[tuple[date, date]],
) -> Iterator[tuple[str, bytes, dict[str, str]]]:
    """Descarga iterativamente los fragmentos de datos de la API de SIMEM para cada ventana.

    Args:
        session: Sesión HTTP activa con configuración de reintentos.
        api: Configuración de la API (URL base, endpoint, timeout).
        dataset_id: Identificador alfanumérico del dataset en SIMEM.
        name: Nombre identificador del dataset.
        windows: Ventanas temporales consecutivas a consultar.

    Yields:
        Tuplas (nombre_archivo, contenido_bytes, metadatos_origen).

    Raises:
        requests.RequestException: Si ocurre un error de conexión, timeout o HTTP 4xx/5xx.

    Examples:
        >>> # Descarga de fragmentos a través de la API REST de SIMEM
    """
    endpoint = api["api_base_url"].rstrip("/") + api["endpoint"]
    timeout = api["timeout_seconds"]

    for start, end in windows:
        params = {"datasetId": dataset_id, "startDate": str(start), "endDate": str(end)}
        filename = f"{name}_{start}_{end}.json"
        logger.debug("Solicitando %s a la API de SIMEM: %s?%s", name, endpoint, params)

        # Ejecución y validación de la solicitud HTTP GET
        try:
            response = session.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "Error en petición HTTP para '%s' (%s a %s): %s",
                name,
                start,
                end,
                exc,
            )
            raise

        yield filename, response.content, {"url": response.url, **params}


def _normalized_source(  # noqa: PLR0913
    chunks: list[pd.DataFrame],
    name: str,
    spec: Dataset,
    manifest: dict,
    revisions: dict[str, str] | None,
    *,
    fallback: dict[str, list[list[str]]] | None = None,
) -> pd.DataFrame:
    """Aplica la normalización a los fragmentos de una fuente usando parámetros del manifiesto.

    Args:
        chunks: Lista de DataFrames con los datos crudos.
        name: Nombre descriptivo de la fuente.
        spec: Especificación de configuración del dataset.
        manifest: Manifiesto en construcción con configuración y ventana temporal.
        revisions: Diccionario de revisiones oficiales asignadas.
        fallback: Diccionario de revisiones alternativas de respaldo.

    Returns:
        DataFrame normalizado y validado.

    Examples:
        >>> # Normalización de fuente según configuración del manifiesto
    """
    settings, window = manifest["settings"], manifest["window"]
    target_value_column = settings["target_column"] if name == "precio_bolsa" else name
    return normalize_dataset(
        chunks,
        spec,
        date.fromisoformat(window["start"]),
        date.fromisoformat(window["end"]),
        settings["timezone"],
        timestamp_column=settings["timestamp_column"],
        value_column=target_value_column,
        revisions=revisions,
        fallback=fallback,
    )


def _stage_chunks(
    stage: Path, artifacts: Iterable[tuple[str, bytes, dict[str, str]]], name: str
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Escribe artefactos en staging, calcula sus hashes SHA-256 y parsea DataFrames.

    Args:
        stage: Ruta al directorio temporal de staging.
        artifacts: Iterable de artefactos (nombre, contenido en bytes, origen).
        name: Nombre identificador del dataset.

    Returns:
        Tupla con (lista_metadatos_archivos, lista_dataframes).

    Raises:
        IngestionError: Si el contenido no se puede parsear a DataFrame.
        OSError: Si falla la escritura en disco.

    Examples:
        >>> # Procesamiento y persistencia en staging con hash SHA-256
    """
    files, chunks = [], []
    for filename, content, origin in artifacts:
        # 1. Parseo de registros en DataFrame
        frame = records_from_bytes(content, Path(filename).suffix)
        chunks.append(frame)

        # 2. Escritura de archivo original en staging
        target_path = stage / filename
        target_path.write_bytes(content)

        # 3. Registro de metadatos criptográficos y dimensiones
        sha256_hash = hashlib.sha256(content).hexdigest()
        files.append(
            {
                "file": filename,
                "sha256": sha256_hash,
                "bytes": len(content),
                "rows": len(frame),
                **origin,
            }
        )
        logger.debug(
            "Fragmento guardado: %s (%d bytes, %d filas, sha256=%s)",
            filename,
            len(content),
            len(frame),
            sha256_hash[:8],
        )
        log_event(logging.INFO, "chunk_ingested", dataset=name, file=filename, rows=len(frame))

    return files, chunks


def _stage_source(stage: Path, artifacts, name: str, spec: Dataset, manifest: dict) -> dict:
    """Gestiona el ciclo de staging, normalización y cálculo de estadísticas para un dataset.

    Args:
        stage: Directorio temporal de staging.
        artifacts: Artefactos descargados u obtenidos localmente.
        name: Nombre del dataset (ej. 'precio_bolsa').
        spec: Especificación de configuración del dataset.
        manifest: Manifiesto en construcción.

    Returns:
        Diccionario con especificación, archivos persistidos y estadísticas de validación.

    Raises:
        IngestionError: Si la normalización o validación de reglas de calidad fallan.

    Examples:
        >>> # Staging y validación de una fuente individual
    """
    logger.info("Iniciando staging y validación para el dataset '%s'", name)
    # 1. Persistencia de fragmentos y cálculo de hashes en staging
    files, chunks = _stage_chunks(stage, artifacts, name)

    # 2. Configuración de revisiones de liquidación y fallback
    calendar = manifest["revision_calendars"].get(name)
    revisions = calendar["revisions"] if calendar is not None else None
    fallback = None
    if calendar is not None and spec.revision_fallback:
        records = pd.concat(_read_snapshot_chunks(stage, calendar["files"]), ignore_index=True)
        fallback = _fallback_revisions(records, spec.revision_as_of)

    # 3. Normalización y verificación de revisiones aplicadas
    normalized = _normalized_source(chunks, name, spec, manifest, revisions, fallback=fallback)
    selected = normalized.attrs["settlement_revisions"]
    if calendar is not None and selected != revisions:
        calendar["preferred_revisions"] = revisions
        calendar["revisions"] = selected
        logger.warning(
            "Uso de fallback de revisiones para '%s': preferidas=%s, seleccionadas=%s",
            name,
            revisions,
            selected,
        )
        log_event(
            logging.WARNING,
            "settlement_revision_fallback",
            dataset=name,
            preferred=revisions,
            selected=selected,
        )

    # 4. Cálculo de estadísticas y verificación de horas completas para precio_bolsa
    stats = {"normalized_rows": len(normalized)}
    if name == "precio_bolsa":
        settings, window = manifest["settings"], manifest["window"]
        start = pd.Timestamp(window["start"], tz=settings["timezone"])
        stop = pd.Timestamp(window["end"], tz=settings["timezone"]) + pd.DateOffset(days=1)
        grid = pd.date_range(start, stop, freq=settings["frequency"], inclusive="left")
        missing = grid.difference(normalized[settings["timestamp_column"]])
        stats.update(expected_hours=len(grid), missing_hours=len(missing))
        if not missing.empty:
            logger.warning(
                "Detectadas %d horas faltantes en precio_bolsa; ejemplos: %s",
                len(missing),
                missing[:3].astype(str).tolist(),
            )
            log_event(
                logging.WARNING,
                "price_hours_missing",
                dataset=name,
                missing_hours=len(missing),
                examples=missing[:3].astype(str).tolist(),
            )

    null_count = int(normalized.iloc[:, -1].isna().sum())
    logger.info(
        "Dataset '%s' validado exitosamente: %d filas normalizadas, %d valores nulos",
        name,
        len(normalized),
        null_count,
    )
    log_event(
        logging.INFO,
        "dataset_validated",
        dataset=name,
        rows=len(normalized),
        null_values=null_count,
    )
    return {"spec": asdict(spec), "files": files, **stats}


def _stage_revision_calendar(  # noqa: PLR0913
    stage: Path,
    session: requests.Session,
    api: dict,
    spec: Dataset,
    manifest: dict,
    *,
    source_dir: Path | None,
    cached: dict[str, list[dict]],
) -> dict:
    """Descarga o reutiliza en caché el calendario oficial de liquidación de revisiones.

    Args:
        stage: Directorio temporal de staging.
        session: Sesión HTTP activa.
        api: Configuración de acceso a la API.
        spec: Especificación del dataset asociado.
        manifest: Manifiesto en construcción.
        source_dir: Directorio local opcional con archivos de exportación.
        cached: Diccionario con calendarios previamente descargados para reutilización.

    Returns:
        Diccionario con metadatos de archivos y el mapeo de revisiones oficiales seleccionadas.

    Raises:
        IngestionError: Si el calendario contiene datos inválidos o inconsistentes.

    Examples:
        >>> # Staging del calendario de revisiones con soporte para caché en memoria
    """
    window = manifest["window"]
    start, end = date.fromisoformat(window["start"]), date.fromisoformat(window["end"])
    name = f"calendario_liquidacion_{spec.revision_calendar_id}"
    logger.debug("Gestionando calendario de liquidación '%s'", name)

    # 1. Recuperación desde caché o descarga de fragmentos
    if name in cached:
        logger.debug("Reutilizando calendario en caché para '%s'", name)
        files = cached[name]
        chunks = _read_snapshot_chunks(stage, files)
    else:
        logger.info("Obteniendo calendario de liquidación '%s'", name)
        days = [month.start_time.date() for month in pd.period_range(start, end, freq="M")]
        artifacts = (
            [_local_chunk(source_dir, name)]
            if source_dir is not None
            else _api_chunks(
                session, api, spec.revision_calendar_id, name, [(day, day) for day in days]
            )
        )
        files, chunks = _stage_chunks(stage, artifacts, name)
        cached[name] = files

    # 2. Resolución de versiones oficiales aplicables al período
    revisions = _official_revisions(
        pd.concat(chunks, ignore_index=True), start, end, spec.revision_as_of
    )
    logger.info(
        "Revisiones oficiales resueltas para dataset '%s': %s",
        spec.dataset_id,
        revisions,
    )
    log_event(
        logging.INFO,
        "settlement_revisions_selected",
        dataset_id=spec.dataset_id,
        revisions=revisions,
    )
    return {"files": files, "revisions": revisions}


def ingest(
    config: dict,
    *,
    source_dir: Path | None = None,
    root: Path = PROJECT_PATH,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Publica un snapshot completo solo si todas las fuentes pasan sus validaciones.

    Coordina la descarga, staging, validación y publicación atómica de todos los
    datasets configurados en la sección 'simem' del archivo de configuración.

    Args:
        config: Diccionario con la configuración completa del sistema.
        source_dir: Directorio local alternativo para carga offline.
        root: Directorio base del proyecto donde se ubica la carpeta 'data'.
        start: Fecha inicial en formato 'AAAA-MM-DD' (opcional, anula config).
        end: Fecha final en formato 'AAAA-MM-DD' (opcional, anula config).

    Returns:
        Ruta Path del directorio del snapshot publicado exitosamente.

    Raises:
        IngestionError: Si la configuración es errónea, faltan fuentes requeridas o
            alguna validación de calidad o consistencia falla.

    Examples:
        >>> snapshot_path = ingest(read_config_mapping(), start="2024-04-01", end="2024-04-02")
        >>> snapshot_path.is_dir()
        True
    """
    logger.info("Iniciando proceso de ingesta de datos de SIMEM")

    # 1. Validación de configuración y parámetros de entrada
    try:
        api, settings = config["simem"], config["data"]
        first = date.fromisoformat(start if start is not None else settings["window"]["start"])
        last = date.fromisoformat(end if end is not None else settings["window"]["end"])
        windows = list(date_chunks(first, last, api["chunk_days"]))
        ZoneInfo(settings["timezone"])
        sources = api["datasets"]
    except KeyError as exc:
        logger.error("Clave requerida ausente en configuración: %s", exc)
        raise IngestionError(f"Configuración incompleta: falta clave {exc}") from exc
    except Exception as exc:
        logger.error("Error al validar configuración de ingesta: %s", exc)
        raise IngestionError(f"Error en configuración de ingesta: {exc}") from exc

    if "precio_bolsa" not in sources:
        logger.error("Falta configurar la fuente esencial 'precio_bolsa'")
        raise IngestionError("Falta configurar la fuente precio_bolsa")
    if api["timeout_seconds"] <= 0 or api["max_retries"] < 0:
        logger.error(
            "Parámetros de red inválidos: timeout_seconds=%s, max_retries=%s",
            api["timeout_seconds"],
            api["max_retries"],
        )
        raise IngestionError("timeout_seconds debe ser positivo y max_retries no negativo")

    # 2. Preparación de rutas de destino y estructura del manifiesto
    captured = datetime.now(UTC)
    parent = root / config["paths"]["data"]["raw"] / "simem" / captured.date().isoformat()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{captured:%H%M%S%f}Z_{uuid4().hex[:8]}"
    manifest = {
        "schema_version": 1,
        "captured_at_utc": captured.isoformat(),
        "window": {"start": str(first), "end": str(last)},
        "settings": {
            key: settings[key]
            for key in ("timezone", "frequency", "timestamp_column", "target_column")
        },
        "datasets": {},
        "revision_calendars": {},
    }

    # 3. Procesamiento en área de staging temporal con publicación atómica
    with (
        tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as temp,
        _session(api["max_retries"]) as session,
    ):
        stage = Path(temp)
        calendars: dict[str, list[dict]] = {}
        for name, mapping in sources.items():
            logger.info("Procesando fuente: '%s'", name)
            spec = Dataset(**mapping)

            # Validación de identificadores y frecuencias
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                logger.error("Nombre de fuente inválido: %s", name)
                raise IngestionError(f"Nombre de fuente inválido: {name}")
            if not re.fullmatch(r"[a-zA-Z0-9]{6}", spec.dataset_id):
                logger.error("%s: dataset_id inválido: %s", name, spec.dataset_id)
                raise IngestionError(f"{name}: dataset_id debe tener seis caracteres alfanuméricos")
            if spec.frequency not in {"1h", "1D"} or not set(spec.group_by) <= set(spec.dimensions):
                logger.error("%s: frecuencia o dimensiones de agrupación inválidas", name)
                raise IngestionError(f"{name}: frecuencia o dimensiones de agrupación inválidas")
            if name == "precio_bolsa" and (
                spec.frequency != settings["frequency"] or spec.dimensions
            ):
                logger.error("Configuración incompatible para precio_bolsa")
                raise IngestionError(
                    "El precio debe ser una serie única con la frecuencia configurada"
                )

            # Procesamiento de calendario de liquidaciones y fragmentos de datos
            try:
                if spec.revision_calendar_id and (
                    not spec.revision
                    or not re.fullmatch(r"[a-zA-Z0-9]{6}", spec.revision_calendar_id)
                ):
                    logger.error(
                        "Configuración de calendario de liquidación inválida para %s", name
                    )
                    raise IngestionError(
                        "El calendario requiere columna de versión e ID de seis caracteres"
                    )
                if spec.revision_as_of is not None and not spec.revision_calendar_id:
                    logger.error("revision_as_of configurado sin revision_calendar_id en %s", name)
                    raise IngestionError("revision_as_of requiere revision_calendar_id")
                if spec.revision_calendar_id:
                    if spec.revision in spec.filters:
                        logger.error("Filtro fijo de versión combinado con calendario en %s", name)
                        raise IngestionError(
                            "No combines un filtro fijo de versión con el calendario"
                        )
                    manifest["revision_calendars"][name] = _stage_revision_calendar(
                        stage, session, api, spec, manifest, source_dir=source_dir, cached=calendars
                    )
                artifacts = (
                    [_local_chunk(source_dir, name)]
                    if source_dir is not None
                    else _api_chunks(session, api, spec.dataset_id, name, windows)
                )
                manifest["datasets"][name] = _stage_source(stage, artifacts, name, spec, manifest)
            except (ValueError, OSError, requests.RequestException) as exc:
                logger.exception(
                    "Error durante la ingesta de la fuente '%s' (%s): %s",
                    name,
                    spec.dataset_id,
                    exc,
                )
                raise IngestionError(f"{name} ({spec.dataset_id}): {exc}") from exc

        # 4. Escritura del manifiesto JSON y renombramiento atómico
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(destination)

    logger.info("Snapshot publicado exitosamente en: %s", destination)
    log_event(logging.INFO, "snapshot_published", path=destination, start=first, end=last)
    return destination


def _read_snapshot_chunks(snapshot: Path, files: list[dict]) -> list[pd.DataFrame]:
    """Lee y valida la integridad criptográfica SHA-256 de los fragmentos de un snapshot.

    Args:
        snapshot: Directorio donde reside el snapshot.
        files: Lista de diccionarios de metadatos de archivos con 'file' y 'sha256'.

    Returns:
        Lista de DataFrames recuperados a partir de los bytes originales.

    Raises:
        IngestionError: Si el hash SHA-256 de algún archivo no coincide con el registrado.
        FileNotFoundError: Si algún archivo registrado no se encuentra en el disco.

    Examples:
        >>> # Lectura verificada por SHA-256
    """
    logger.debug("Leyendo y verificando %d archivos en snapshot '%s'", len(files), snapshot)
    chunks = []
    for entry in files:
        path = snapshot / entry["file"]
        if not path.is_file():
            logger.error("Archivo no encontrado en snapshot: %s", path)
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.error("Error de E/S al leer fragmento '%s': %s", path, exc)
            raise IngestionError(f"Error al leer fragmento {path}: {exc}") from exc

        current_hash = hashlib.sha256(content).hexdigest()
        if current_hash != entry["sha256"]:
            logger.error(
                "Discrepancia de SHA-256 en %s: esperado=%s, actual=%s",
                path,
                entry["sha256"],
                current_hash,
            )
            raise IngestionError(f"SHA-256 no coincide: {path}")
        chunks.append(records_from_bytes(content, path.suffix))
    return chunks


def read_snapshot(snapshot: Path, name: str) -> pd.DataFrame:
    """Reconstruye una serie con las reglas y el calendario congelados en su manifiesto.

    Args:
        snapshot: Ruta del directorio del snapshot publicado.
        name: Nombre identificador del dataset a reconstruir.

    Returns:
        DataFrame con la serie de tiempo validada y normalizada según el manifiesto.

    Raises:
        IngestionError: Si el manifiesto no existe, si la versión del esquema no es compatible,
            si el dataset no está registrado o si las revisiones no coinciden con las congeladas.

    Examples:
        >>> # Reconstrucción reproducible de un dataset congelado en snapshot
    """
    logger.info("Reconstruyendo serie '%s' desde snapshot '%s'", name, snapshot)

    # 1. Lectura y validación del manifiesto del snapshot
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        logger.error("Manifiesto no encontrado en el snapshot: %s", manifest_path)
        raise IngestionError(f"Manifiesto no encontrado: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error al leer el manifiesto del snapshot: %s", exc)
        raise IngestionError(f"Error al leer manifiesto: {exc}") from exc

    if manifest.get("schema_version") != 1:
        logger.error(
            "Versión de esquema del manifiesto no soportada: %s",
            manifest.get("schema_version"),
        )
        raise IngestionError("Versión de manifiesto no soportada")

    if name not in manifest.get("datasets", {}):
        logger.error("El dataset '%s' no está registrado en el manifiesto", name)
        raise IngestionError(f"Dataset '{name}' no presente en el manifiesto")

    # 2. Carga de especificación y resolución de calendarios de revisión
    source = manifest["datasets"][name]
    spec = Dataset(**source["spec"])
    revisions = None
    fallback = None

    if spec.revision_calendar_id:
        calendar = manifest["revision_calendars"][name]
        window = manifest["window"]
        records = pd.concat(_read_snapshot_chunks(snapshot, calendar["files"]), ignore_index=True)
        revisions = _official_revisions(
            records,
            date.fromisoformat(window["start"]),
            date.fromisoformat(window["end"]),
            spec.revision_as_of,
        )
        if spec.revision_fallback:
            fallback = _fallback_revisions(records, spec.revision_as_of)

    # 3. Normalización y verificación de coincidencia con revisiones congeladas
    result = _normalized_source(
        _read_snapshot_chunks(snapshot, source["files"]),
        name,
        spec,
        manifest,
        revisions,
        fallback=fallback,
    )
    if spec.revision_calendar_id and result.attrs["settlement_revisions"] != calendar["revisions"]:
        logger.error(
            "Discrepancia en revisiones reconstruidas (%s) vs congeladas (%s)",
            result.attrs["settlement_revisions"],
            calendar["revisions"],
        )
        raise IngestionError("Las revisiones reconstruidas no coinciden con el manifiesto")

    logger.info("Serie '%s' reconstruida exitosamente (%d filas)", name, len(result))
    return result


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos para la ingesta de datos de SIMEM.

    Args:
        argv: Lista opcional de argumentos de línea de comandos. Si es None, utiliza sys.argv[1:].

    Returns:
        0 si la ingesta fue exitosa, 1 en caso de error o excepción.

    Examples:
        >>> exit_code = main(["--start", "2024-04-01", "--end", "2024-04-02"])
        >>> isinstance(exit_code, int)
        True
    """
    # 1. Configuración y parseo de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start", help="Fecha inicial inclusiva (AAAA-MM-DD)")
    parser.add_argument("--end", help="Fecha final inclusiva (AAAA-MM-DD)")
    parser.add_argument("--local-dir", type=Path, help="Directorio con exportaciones de SIMEM")
    args = parser.parse_args(argv)

    # 2. Ejecución protegida de la ingesta y captura de excepciones
    try:
        logger.info("Ejecutando ingesta desde CLI con configuración: %s", args.config)
        ingest(
            read_config_mapping(args.config),
            source_dir=args.local_dir,
            start=args.start,
            end=args.end,
        )
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        logger.exception("Fallo durante la ejecución de ingesta en main: %s", exc)
        log_event(logging.ERROR, "ingestion_failed", error=str(exc))
        return 1

    logger.info("Ingesta finalizada con éxito")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
