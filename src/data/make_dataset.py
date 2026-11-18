"""Ingesta reproducible de SIMEM: python -m src.data.make_dataset --help."""

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

MAX_CHUNK_DAYS = 31


class IngestionError(ValueError):
    """Error de configuración, contrato o calidad de la fuente."""


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    timestamp: str
    value: str
    frequency: str
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    aggregate: bool = False
    revision: str | None = None


def date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """Divide un rango inclusivo en ventanas consecutivas sin solapamiento."""
    if start > end or not 1 <= days <= MAX_CHUNK_DAYS:
        raise IngestionError(f"Rango invertido o chunk_days fuera de [1, {MAX_CHUNK_DAYS}]")
    while start <= end:
        last = start + timedelta(days=min(days - 1, (end - start).days))
        yield start, last
        if last == end:
            break
        start = last + timedelta(days=1)


def records_from_bytes(content: bytes, suffix: str) -> pd.DataFrame:
    """Lee un CSV estructurado o la respuesta JSON de PublicData."""
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content), dtype="string", keep_default_na=False, na_values=[""])

    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise IngestionError("SIMEM no devolvió success=true")
    result = payload.get("result")
    if result is None and payload.get("hasData") is False:
        return pd.DataFrame()
    records = result.get("records") if isinstance(result, dict) else None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise IngestionError("SIMEM: se esperaba una lista de objetos en result.records")
    return pd.DataFrame.from_records(records)


def _normalize_chunk(raw: pd.DataFrame, spec: Dataset, timezone: str) -> pd.DataFrame:
    required = {spec.timestamp, spec.value, *spec.filters, *spec.dimensions}
    if spec.revision:
        required.add(spec.revision)
    if missing := required - set(raw.columns):
        raise IngestionError(f"Columnas de origen faltantes: {sorted(missing)}")
    for column, choices in spec.filters.items():
        raw = raw.loc[raw[column].isin([choices] if isinstance(choices, str) else choices)]

    timestamps = pd.to_datetime(raw[spec.timestamp], errors="raise", format="ISO8601")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone)
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    if timestamps.isna().any() or not timestamps.eq(timestamps.dt.floor(spec.frequency)).all():
        raise IngestionError(f"Fechas nulas o fuera de la malla {spec.frequency}")

    values = pd.to_numeric(raw[spec.value], errors="raise").astype("float64")
    if np.isinf(values).any():
        raise IngestionError("Valores infinitos en la columna numérica")
    clean = raw[spec.dimensions].astype("string").assign(fecha=timestamps, valor=values)
    if spec.revision:
        clean["revision"] = raw[spec.revision].astype("string")
    if clean[spec.dimensions].isna().any().any():
        raise IngestionError("Dimensiones nulas en la clave temporal")
    return clean


def normalize_dataset(  # noqa: PLR0913
    chunks: list[pd.DataFrame],
    spec: Dataset,
    start: date,
    end: date,
    timezone: str,
    *,
    timestamp_column: str = "fecha",
    value_column: str = "precio_bolsa",
) -> pd.DataFrame:
    """Valida las claves naturales antes de construir series a su granularidad nativa."""
    blocks = [_normalize_chunk(chunk, spec, timezone) for chunk in chunks if len(chunk.index)]
    if not blocks:
        raise IngestionError("Fuente sin registros")
    data = pd.concat(blocks, ignore_index=True)
    data = data.loc[data["fecha"].dt.date.between(start, end)]
    if data.empty or not data["valor"].notna().any():
        raise IngestionError("Sin valores en la ventana para los filtros seleccionados")
    if spec.revision and (
        data["revision"].isna().any() or data.groupby("fecha")["revision"].nunique().gt(1).any()
    ):
        raise IngestionError("Versiones de liquidación nulas o múltiples en la misma fecha/hora")

    natural_key = ["fecha", *spec.dimensions]
    if data.duplicated(natural_key).any():
        raise IngestionError(f"Duplicados en la clave temporal {natural_key}")
    key = natural_key
    if spec.aggregate:
        key = ["fecha", *spec.group_by]
        grouped = data.groupby(key, sort=False, dropna=False)["valor"]
        totals = grouped.sum().where(grouped.count().eq(grouped.size()))
        data = totals.reset_index()
    if not data["valor"].notna().any():
        raise IngestionError("Todos los valores agregados son faltantes")
    return (
        data.sort_values(key)[[*key, "valor"]]
        .rename(columns={"fecha": timestamp_column, "valor": value_column})
        .reset_index(drop=True)
    )


def _session(max_retries: int) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _local_chunk(source_dir: Path, name: str) -> tuple[str, bytes, dict[str, str]]:
    candidates = [
        path for suffix in (".csv", ".json") if (path := source_dir / f"{name}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise IngestionError(f"Se requiere exactamente un archivo {name}.csv o {name}.json")
    path = candidates[0]
    return path.name, path.read_bytes(), {"source": str(path.resolve())}


def _api_chunks(session, api, spec, name, windows) -> Iterator[tuple[str, bytes, dict[str, str]]]:
    endpoint = api["api_base_url"].rstrip("/") + api["endpoint"]
    for start, end in windows:
        params = {"datasetId": spec.dataset_id, "startDate": str(start), "endDate": str(end)}
        response = session.get(endpoint, params=params, timeout=api["timeout_seconds"])
        response.raise_for_status()
        yield f"{name}_{start}_{end}.json", response.content, {"url": response.url, **params}


def _normalized_source(chunks: list[pd.DataFrame], name: str, spec: Dataset, manifest: dict):
    settings, window = manifest["settings"], manifest["window"]
    return normalize_dataset(
        chunks,
        spec,
        date.fromisoformat(window["start"]),
        date.fromisoformat(window["end"]),
        settings["timezone"],
        timestamp_column=settings["timestamp_column"],
        value_column=settings["target_column"] if name == "precio_bolsa" else name,
    )


def _stage_source(
    stage: Path,
    artifacts: Iterable[tuple[str, bytes, dict[str, str]]],
    name: str,
    spec: Dataset,
    manifest: dict,
) -> dict:
    files, chunks = [], []
    for filename, content, origin in artifacts:
        frame = records_from_bytes(content, Path(filename).suffix)
        chunks.append(frame)
        (stage / filename).write_bytes(content)
        files.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "rows": len(frame),
                **origin,
            }
        )
        log_event(logging.INFO, "chunk_ingested", dataset=name, file=filename, rows=len(frame))
    normalized = _normalized_source(chunks, name, spec, manifest)
    log_event(
        logging.INFO,
        "dataset_validated",
        dataset=name,
        rows=len(normalized),
        null_values=int(normalized.iloc[:, -1].isna().sum()),
    )
    return {"spec": asdict(spec), "files": files, "normalized_rows": len(normalized)}


def ingest(
    config: dict,
    *,
    source_dir: Path | None = None,
    root: Path = PROJECT_PATH,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Publica un snapshot completo solo si todas las fuentes pasan sus validaciones."""
    api, settings = config["simem"], config["data"]
    first = date.fromisoformat(start if start is not None else settings["window"]["start"])
    last = date.fromisoformat(end if end is not None else settings["window"]["end"])
    windows = list(date_chunks(first, last, api["chunk_days"]))
    ZoneInfo(settings["timezone"])
    sources = api["datasets"]
    if "precio_bolsa" not in sources:
        raise IngestionError("Falta configurar la fuente precio_bolsa")
    if api["timeout_seconds"] <= 0 or api["max_retries"] < 0:
        raise IngestionError("timeout_seconds debe ser positivo y max_retries no negativo")

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
    }

    with (
        tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as temp,
        _session(api["max_retries"]) as session,
    ):
        stage = Path(temp)
        for name, mapping in sources.items():
            spec = Dataset(**mapping)
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise IngestionError(f"Nombre de fuente inválido: {name}")
            if not re.fullmatch(r"[a-zA-Z0-9]{6}", spec.dataset_id):
                raise IngestionError(f"{name}: dataset_id debe tener seis caracteres alfanuméricos")
            if spec.frequency not in {"1h", "1D"} or not set(spec.group_by) <= set(spec.dimensions):
                raise IngestionError(f"{name}: frecuencia o dimensiones de agrupación inválidas")
            if name == "precio_bolsa" and (
                spec.frequency != settings["frequency"] or spec.dimensions
            ):
                raise IngestionError(
                    "El precio debe ser una serie única con la frecuencia configurada"
                )
            try:
                artifacts = (
                    [_local_chunk(source_dir, name)]
                    if source_dir is not None
                    else _api_chunks(session, api, spec, name, windows)
                )
                manifest["datasets"][name] = _stage_source(stage, artifacts, name, spec, manifest)
            except (ValueError, OSError, requests.RequestException) as exc:
                raise IngestionError(f"{name} ({spec.dataset_id}): {exc}") from exc
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(destination)
    log_event(logging.INFO, "snapshot_published", path=destination, start=first, end=last)
    return destination


def read_snapshot(snapshot: Path, name: str) -> pd.DataFrame:
    """Reconstruye una serie con las reglas congeladas en su manifiesto."""
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise IngestionError("Versión de manifiesto no soportada")
    source = manifest["datasets"][name]
    chunks = []
    for entry in source["files"]:
        path = snapshot / entry["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise IngestionError(f"SHA-256 no coincide: {path}")
        chunks.append(records_from_bytes(content, path.suffix))
    return _normalized_source(chunks, name, Dataset(**source["spec"]), manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start", help="Fecha inicial inclusiva (AAAA-MM-DD)")
    parser.add_argument("--end", help="Fecha final inclusiva (AAAA-MM-DD)")
    parser.add_argument("--local-dir", type=Path, help="Directorio con exportaciones de SIMEM")
    args = parser.parse_args(argv)
    try:
        ingest(
            read_config_mapping(args.config),
            source_dir=args.local_dir,
            start=args.start,
            end=args.end,
        )
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        log_event(logging.ERROR, "ingestion_failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
