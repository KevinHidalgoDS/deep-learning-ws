#!/usr/bin/env python
# -*- coding: utf-8 -*-  # noqa: UP009

"""Utilidades de auditoría y análisis exploratorio de datos (EDA) del precio horario del MEM.

Este módulo provee herramientas para consultar la API de XM (Mercado de Energía Mayorista
de Colombia), auditar la calidad, completitud y tipos de datos en DataFrames, pivotear
series de formato ancho a formato largo con localización horaria en Bogotá, y calcular
estadísticas descriptivas robustas con detección de valores atípicos mediante el rango
intercuartílico (IQR).

Proyecto: deep-learning-ws

Programa: price_eda.py

Soporte: kfhidalgoh@unal.edu.co

Versión: 1.0.0

Lenguaje: Python 3.14.5

CD: 20261118

LUD: 20261118

Comentarios:
    - 2026-11-18 Kevin Hidalgo -> creación.

Ejemplo:
    Para auditar un DataFrame y transformar una serie horaria::

        from datetime import date
        from src.data.price_eda import audit_dataframe, melt_hourly_to_long

        resumen = audit_dataframe(df_precios, "Precio_Bolsa")
        df_largo = melt_hourly_to_long(df_precios, "Precio")
"""

__authors__ = ["Kevin Hidalgo"]
__contact__ = "kfhidalgoh@unal.edu.co"
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Kevin Hidalgo"]
__email__ = "kfhidalgoh@unal.edu.co"
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-18"

import asyncio
import logging
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.utils.logger import log_event

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CONSTANTES Y LOGGER DEL MÓDULO
# -----------------------------------------------------------------------------
XM_HOURS = tuple(f"Values_Hour{hour:02d}" for hour in range(1, 25))

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


def request_xm_price(start: date, end: date) -> pd.DataFrame:
    """Consulta el precio de bolsa nacional en XM mediante transporte compatible con Python 3.14.

    Ejecuta una petición a la API de XM para obtener la métrica 'PrecBolsNaci' del sistema
    nacional en un rango de fechas determinado, utilizando un adaptador asíncrono en hilo
    independiente para compatibilidad con la biblioteca pydataxm.

    Args:
        start: Fecha inicial inclusiva de la consulta.
        end: Fecha final inclusiva de la consulta.

    Returns:
        DataFrame con los registros de precios horarios suministrados por la API de XM.

    Raises:
        ValueError: Si la fecha inicial es posterior a la final, o si la API de XM no
            retorna registros para el intervalo solicitado.
        RuntimeError: Si ocurre un error de comunicación HTTP o falla la librería pydataxm.

    Examples:
        >>> from datetime import date
        >>> # df_precios = request_xm_price(date(2025, 1, 1), date(2025, 1, 2))
    """
    # 1. Validación de precondiciones de rango de fechas
    logger.debug("Iniciando solicitud de precios XM desde %s hasta %s", start, end)
    if start > end:
        logger.error("Rango de fechas inválido: start=%s es posterior a end=%s", start, end)
        raise ValueError(
            f"Rango de fechas invertido: fecha inicial {start} posterior a final {end}."
        )

    # 2. Configuración del lector adaptado de pydataxm
    try:
        from pydataxm.pydataxm import ReadDB  # noqa: PLC0415
    except ImportError as exc:
        logger.exception("Error al importar la biblioteca pydataxm: %s", exc)
        raise RuntimeError(f"Biblioteca 'pydataxm' no disponible en el entorno: {exc}") from exc

    class HourlyReader(ReadDB):
        """Adaptador de ReadDB para evitar conflictos de bucle de eventos en Python 3.14."""

        async def async_get_df(self, body: dict, endpoint: str) -> pd.DataFrame:
            # aiohttp no reconoce las tareas parcheadas por nest_asyncio en Python 3.14.
            logger.debug("Enviando petición HTTP POST a endpoint XM: %s", endpoint)
            try:
                response = await asyncio.to_thread(requests.post, self.url, json=body, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as req_exc:
                logger.error("Fallo de conexión o respuesta HTTP de XM: %s", req_exc)
                raise
            except ValueError as json_exc:
                logger.error("Respuesta no válida de XM (no decodifica como JSON): %s", json_exc)
                raise

            if "Items" not in payload:
                logger.error("Estructura de respuesta inesperada: clave 'Items' ausente")
                raise ValueError("Respuesta de XM no contiene la clave 'Items'")

            return pd.json_normalize(payload["Items"], endpoint, "Date", sep="_")

    # 3. Ejecución de la consulta de datos en XM
    try:
        reader = HourlyReader()
        price = reader.request_data("PrecBolsNaci", "Sistema", start, end)
    except Exception as exc:
        logger.exception("Excepción durante la consulta de precios a XM: %s", exc)
        raise

    # 4. Verificación de completitud de la respuesta y logging
    if price.empty:
        logger.error(
            "La consulta a XM no retornó registros de PrecBolsNaci para %s a %s", start, end
        )
        raise ValueError(f"XM no devolvió datos de PrecBolsNaci entre {start} y {end}.")

    logger.info("Precios recibidos de XM exitosamente: %d registros obtenidos", len(price))
    log_event(logging.INFO, "xm_price_received", start=start, end=end, rows=len(price))
    return price


def audit_dataframe(df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
    """Resume tipos de datos, completitud y cardinalidad por columna sin modificar los datos.

    Calcula métricas diagnósticas clave para cada columna de un DataFrame, incluyendo
    conteo de no nulos, nulos absolutos, porcentaje de nulos y conteo de valores únicos.

    Args:
        df: DataFrame de pandas a auditar.
        dataset_name: Nombre descriptivo o identificador del conjunto de datos.

    Returns:
        DataFrame con las métricas de calidad de datos organizadas por columna.

    Raises:
        TypeError: Si el argumento `df` proporcionado no es un DataFrame de pandas.

    Examples:
        >>> import pandas as pd
        >>> df_test = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "x"]})
        >>> auditoria = audit_dataframe(df_test, "Prueba")
        >>> list(auditoria["Columna"])
        ['A', 'B']
    """
    # 1. Validación de tipo del argumento de entrada
    if not isinstance(df, pd.DataFrame):
        logger.error("Tipo de dato inválido en audit_dataframe: se esperaba pd.DataFrame")
        raise TypeError("Se esperaba un objeto pandas DataFrame para la auditoría.")

    # 2. Registro de métricas generales del conjunto de datos
    filas, columnas = df.shape
    duplicados = int(df.duplicated().sum())

    logger.info(
        "Auditoría ejecutada para '%s': %d filas, %d columnas, %d duplicados",
        dataset_name,
        filas,
        columnas,
        duplicados,
    )
    if df.empty:
        logger.warning("El conjunto de datos '%s' está vacío (0 registros)", dataset_name)
    if duplicados > 0:
        logger.warning(
            "Se detectaron %d filas duplicadas en el conjunto de datos '%s'",
            duplicados,
            dataset_name,
        )

    log_event(
        logging.INFO,
        "dataframe_audited",
        dataset=dataset_name,
        rows=filas,
        columns=columnas,
        duplicate_rows=duplicados,
    )

    # 3. Construcción del resumen diagnóstico por columna
    logger.debug("Calculando métricas diagnósticas por columna para '%s'", dataset_name)
    return pd.DataFrame(
        {
            "Columna": df.columns,
            "Tipo_Dato": df.dtypes.astype(str).to_numpy(),
            "Registros": len(df),
            "No_Nulos": df.count().to_numpy(),
            "Nulos": df.isna().sum().to_numpy(),
            "Pct_Nulos": df.isna().mean().mul(100).round(2).to_numpy(),
            "Valores_Unicos": df.nunique().to_numpy(),
        }
    )


def melt_hourly_to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Convierte una tabla horaria en formato ancho de XM a formato largo con zona horaria Bogotá.

    Transforma las columnas 'Values_Hour01' a 'Values_Hour24' a marcas de tiempo continuas,
    mapeando Hour01 a las 00:00 y Hour24 a las 23:00 en la zona horaria 'America/Bogota'.

    Args:
        df: DataFrame de XM en formato ancho que contiene la columna 'Date' y 'Values_Hour01'..'24'.
        metric_name: Nombre asignado a la columna resultante que almacena los valores numéricos.

    Returns:
        DataFrame en formato largo ordenado por Timestamp con columnas:
        ['Timestamp', 'Date', 'Hour_Int', <metric_name>].

    Raises:
        TypeError: Si `df` no es un DataFrame de pandas.
        ValueError: Si la tabla está vacía, tiene columnas duplicadas, carece de columnas
            horarias requeridas, utiliza un nombre reservado para la métrica, contiene fechas
            inválidas/duplicadas o valores infinitos en la métrica.

    Examples:
        >>> # df_largo = melt_hourly_to_long(df_xm, "Precio")
        >>> # df_largo.columns.tolist() == ["Timestamp", "Date", "Hour_Int", "Precio"]
    """
    # 1. Validación de precondiciones estructurales y nombres de columnas
    if not isinstance(df, pd.DataFrame):
        logger.error("Argumento 'df' no es un DataFrame en melt_hourly_to_long")
        raise TypeError("El argumento 'df' debe ser un DataFrame de pandas.")

    if not df.columns.is_unique or df.empty:
        logger.error("DataFrame vacío o con nombres de columnas duplicados")
        raise ValueError("Se requiere una tabla XM no vacía, con nombres de columnas únicos.")

    if missing := {"Date", *XM_HOURS} - set(df.columns):
        logger.error("Columnas obligatorias de XM faltantes: %s", sorted(missing))
        raise ValueError(f"Columnas de XM faltantes: {sorted(missing)}")

    reserved_names = {"Timestamp", "Date", "Hour_Int", "Hour_Label", *XM_HOURS}
    if metric_name in reserved_names:
        logger.error("El nombre de métrica '%s' colisiona con nombres reservados", metric_name)
        raise ValueError(f"Nombre de métrica reservado: {metric_name}")

    logger.debug("Iniciando transformación de ancho a largo para la métrica '%s'", metric_name)

    # 2. Conversión, localización y verificación de marcas de fecha
    try:
        dates = pd.to_datetime(df["Date"], format="ISO8601", errors="raise")
    except Exception as exc:
        logger.error("Error al parsear fechas en columna 'Date': %s", exc)
        raise ValueError(f"Error al procesar la columna 'Date': {exc}") from exc

    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("America/Bogota")
    else:
        dates = dates.dt.tz_convert("America/Bogota")

    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        logger.error("Fechas nulas o con componente horaria no nula en la columna 'Date'")
        raise ValueError("Date debe contener fechas locales válidas sin componente horaria.")

    if dates.duplicated().any():
        logger.error("Se encontraron fechas duplicadas en los registros nacionales de XM")
        raise ValueError("XM devolvió más de un registro nacional para una misma fecha.")

    # 3. Despivotado (melt) de columnas horarias
    data = df.loc[:, list(XM_HOURS)].assign(Date=dates)
    long = data.melt(id_vars="Date", var_name="Hour_Label", value_name=metric_name)

    # 4. Cálculo de marcas temporales precisas y tipado numérico
    long["Hour_Int"] = long.pop("Hour_Label").str[-2:].astype(int)
    long["Timestamp"] = long["Date"] + pd.to_timedelta(long["Hour_Int"] - 1, unit="h")

    try:
        long[metric_name] = pd.to_numeric(long[metric_name], errors="raise").astype("float64")
    except Exception as exc:
        logger.error("Error al convertir la métrica '%s' a float64: %s", metric_name, exc)
        raise ValueError(
            f"Valores no numéricos detectados en la métrica '{metric_name}': {exc}"
        ) from exc

    # 5. Validación de valores numéricos finitos y ordenamiento final
    if np.isinf(long[metric_name]).any():
        logger.error("Se detectaron valores infinitos en la métrica '%s'", metric_name)
        raise ValueError("XM devolvió precios infinitos.")

    result = long.sort_values("Timestamp", ignore_index=True)[
        ["Timestamp", "Date", "Hour_Int", metric_name]
    ]
    logger.info(
        "Transformación a formato largo completada exitosamente: %d observaciones generadas",
        len(result),
    )
    return result


def compute_extended_statistics(series: pd.Series, variable_name: str = "Variable") -> pd.Series:
    """Calcula estadísticas descriptivas extendidas y conteo de valores atípicos mediante Tukey.

    Calcula medidas de posición, dispersión, forma (asimetría y curtosis) y umbrales
    de valores atípicos basados en 1.5 veces el rango intercuartílico (IQR).

    Args:
        series: Serie numérica de pandas a analizar.
        variable_name: Etiqueta o nombre identificador de la variable.

    Returns:
        Serie de pandas con todas las métricas estadísticas calculadas.

    Raises:
        TypeError: Si el argumento `series` no es una instancia de `pd.Series`.
        ValueError: Si `variable_name` no es una cadena válida.

    Examples:
        >>> import pandas as pd
        >>> datos = pd.Series([10.0, 12.0, 11.5, 14.0, 100.0])
        >>> estadisticas = compute_extended_statistics(datos, "Precio")
        >>> estadisticas["Outliers_IQR_Count"]
        1
    """
    # 1. Validación de tipo del argumento de entrada
    if not isinstance(series, pd.Series):
        logger.error(
            "Tipo de entrada inválido en compute_extended_statistics: se requiere pd.Series"
        )
        raise TypeError("Se esperaba una serie de pandas (pd.Series) como entrada.")

    if not isinstance(variable_name, str) or not variable_name.strip():
        logger.error("Nombre de variable inválido en compute_extended_statistics")
        raise ValueError("El nombre de la variable debe ser una cadena de texto no vacía.")

    logger.debug("Calculando estadísticas extendidas para la variable '%s'", variable_name)

    # 2. Limpieza de datos nulos y cálculo de cuantiles
    clean = series.dropna()
    total_obs = len(clean)

    if total_obs == 0:
        logger.warning(
            "La serie '%s' no contiene observaciones válidas tras descartar nulos",
            variable_name,
        )

    q1, median, q3 = clean.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1

    # 3. Determinación de umbrales y conteo de valores atípicos (Tukey IQR)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers_mask = clean.lt(lower) | clean.gt(upper)
    outlier_count = int(outliers_mask.sum())

    if outlier_count > 0:
        logger.debug(
            "Variable '%s': %d valores atípicos detectados fuera de [%.2f, %.2f]",
            variable_name,
            outlier_count,
            lower,
            upper,
        )

    # 4. Cálculo de momentos estadísticos (media, desviación, asimetría, curtosis)
    mean = clean.mean()
    std = clean.std()
    cv_pct = 100 * std / mean if pd.notna(mean) and mean > 0 else np.nan
    outlier_pct = 100 * outlier_count / total_obs if total_obs > 0 else np.nan

    # 5. Estructuración y retorno de la serie de resultados
    logger.info(
        "Estadísticas calculadas para '%s': %d observaciones, media=%.2f, IQR=%.2f",
        variable_name,
        total_obs,
        mean if pd.notna(mean) else 0.0,
        iqr if pd.notna(iqr) else 0.0,
    )
    return pd.Series(
        {
            "Variable": variable_name,
            "Observaciones": total_obs,
            "Media": mean,
            "Desv_Estandar": std,
            "Coef_Variacion_%": cv_pct,
            "Minimo": clean.min(),
            "Q1 (25%)": q1,
            "Mediana (50%)": median,
            "Q3 (75%)": q3,
            "Maximo": clean.max(),
            "IQR": iqr,
            "Asimetria": clean.skew(),
            "Curtosis": clean.kurtosis(),
            "Outliers_IQR_Count": outlier_count,
            "Outliers_IQR_%": outlier_pct,
            "Umbral_Inf_IQR": lower,
            "Umbral_Sup_IQR": upper,
        }
    )
