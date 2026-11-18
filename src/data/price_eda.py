"""Utilidades compartidas para auditar el precio horario del MEM."""

import asyncio
import logging
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.utils.logger import log_event

XM_HOURS = tuple(f"Values_Hour{hour:02d}" for hour in range(1, 25))


def request_xm_price(start: date, end: date) -> pd.DataFrame:
    """Consulta el precio con ReadDB y un transporte compatible con Python 3.14."""
    from pydataxm.pydataxm import ReadDB  # noqa: PLC0415

    class HourlyReader(ReadDB):
        async def async_get_df(self, body: dict, endpoint: str) -> pd.DataFrame:
            # aiohttp no reconoce las tareas que nest_asyncio parchea en Python 3.14.
            response = await asyncio.to_thread(requests.post, self.url, json=body, timeout=30)
            response.raise_for_status()
            return pd.json_normalize(response.json()["Items"], endpoint, "Date", sep="_")

    price = HourlyReader().request_data("PrecBolsNaci", "Sistema", start, end)
    if price.empty:
        raise ValueError(f"XM no devolvió datos de PrecBolsNaci entre {start} y {end}.")
    log_event(logging.INFO, "xm_price_received", start=start, end=end, rows=len(price))
    return price


def audit_dataframe(df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
    """Resume tipos, completitud y cardinalidad por columna, sin modificar los datos."""
    log_event(
        logging.INFO,
        "dataframe_audited",
        dataset=dataset_name,
        rows=df.shape[0],
        columns=df.shape[1],
        duplicate_rows=int(df.duplicated().sum()),
    )
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
    """Convierte una serie XM nacional: Hour01 → 00:00 y Hour24 → 23:00 de Bogotá."""
    if not df.columns.is_unique or df.empty:
        raise ValueError("Se requiere una tabla XM no vacía, con nombres de columnas únicos.")
    if missing := {"Date", *XM_HOURS} - set(df.columns):
        raise ValueError(f"Columnas de XM faltantes: {sorted(missing)}")
    if metric_name in {"Timestamp", "Date", "Hour_Int", "Hour_Label", *XM_HOURS}:
        raise ValueError(f"Nombre de métrica reservado: {metric_name}")

    dates = pd.to_datetime(df["Date"], format="ISO8601", errors="raise")
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("America/Bogota")
    else:
        dates = dates.dt.tz_convert("America/Bogota")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Date debe contener fechas locales válidas sin componente horaria.")
    if dates.duplicated().any():
        raise ValueError("XM devolvió más de un registro nacional para una misma fecha.")

    data = df.loc[:, list(XM_HOURS)].assign(Date=dates)
    long = data.melt(id_vars="Date", var_name="Hour_Label", value_name=metric_name)
    long["Hour_Int"] = long.pop("Hour_Label").str[-2:].astype(int)
    long["Timestamp"] = long["Date"] + pd.to_timedelta(long["Hour_Int"] - 1, unit="h")
    long[metric_name] = pd.to_numeric(long[metric_name], errors="raise").astype("float64")
    if np.isinf(long[metric_name]).any():
        raise ValueError("XM devolvió precios infinitos.")
    return long.sort_values("Timestamp", ignore_index=True)[
        ["Timestamp", "Date", "Hour_Int", metric_name]
    ]


def compute_extended_statistics(series: pd.Series, variable_name: str = "Variable") -> pd.Series:
    """Calcula estadísticas descriptivas y atípicos globales de Tukey (1,5 veces el IQR)."""
    clean = series.dropna()
    q1, median, q3 = clean.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int((clean.lt(lower) | clean.gt(upper)).sum())
    mean, std = clean.mean(), clean.std()
    return pd.Series(
        {
            "Variable": variable_name,
            "Observaciones": len(clean),
            "Media": mean,
            "Desv_Estandar": std,
            "Coef_Variacion_%": 100 * std / mean if pd.notna(mean) and mean > 0 else np.nan,
            "Minimo": clean.min(),
            "Q1 (25%)": q1,
            "Mediana (50%)": median,
            "Q3 (75%)": q3,
            "Maximo": clean.max(),
            "IQR": iqr,
            "Asimetria": clean.skew(),
            "Curtosis": clean.kurtosis(),
            "Outliers_IQR_Count": count,
            "Outliers_IQR_%": 100 * count / len(clean) if len(clean) else np.nan,
            "Umbral_Inf_IQR": lower,
            "Umbral_Sup_IQR": upper,
        }
    )
