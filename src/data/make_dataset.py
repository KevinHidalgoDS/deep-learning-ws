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
    revision_calendar_id: str | None = None
    revision_as_of: str | None = None
    revision_fallback: bool = False


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
        clean["revision"] = raw[spec.revision].astype("string").str.strip()
    if clean[spec.dimensions].isna().any().any():
        raise IngestionError("Dimensiones nulas en la clave temporal")
    return clean


def _official_revisions(
    calendar: pd.DataFrame, start: date, end: date, as_of: str | None
) -> dict[str, str]:
    required = {"Version", "FechaInicio", "FechaFin", "FechaPublicacion", "EsMaximaVersion"}
    if missing := required - set(calendar.columns):
        raise IngestionError(f"Columnas faltantes en calendario de liquidación: {sorted(missing)}")
    data = calendar[list(required)].drop_duplicates().copy()
    dates = ["FechaInicio", "FechaFin", "FechaPublicacion"]
    data[dates] = data[dates].apply(pd.to_datetime, format="%Y-%m-%d", errors="raise")
    data["Version"] = data["Version"].astype("string").str.strip()
    if data[dates].isna().any().any() or data["Version"].fillna("").eq("").any():
        raise IngestionError("El calendario contiene fechas o versiones de liquidación nulas")

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
            eligible = rows.loc[
                rows["FechaPublicacion"].le(pd.Timestamp(date.fromisoformat(as_of)))
            ]
            eligible = eligible.loc[
                eligible["FechaPublicacion"].eq(eligible["FechaPublicacion"].max())
            ]
        if len(eligible) != 1:
            raise IngestionError(
                f"Se esperaba una revisión oficial única para {period}"
                f" (corte: {as_of or 'vigente'}); encontradas: {len(eligible)}"
            )
        revisions[str(period)] = eligible["Version"].iloc[0]
    return revisions


def _fallback_revisions(calendar: pd.DataFrame, as_of: str | None) -> dict[str, list[list[str]]]:
    published = pd.to_datetime(calendar["FechaPublicacion"])
    rows = calendar.assign(published=published, period=calendar["FechaInicio"].str[:7])
    if as_of is not None:
        rows = rows.loc[rows["published"].le(pd.Timestamp(as_of))]
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
    distinct = data[["fecha", "revision"]].drop_duplicates()
    distinct = distinct.assign(period=distinct["fecha"].dt.strftime("%Y-%m"))
    selected = preferred.copy()
    for period, month in distinct.groupby("period"):
        counts = month.groupby("revision")["fecha"].nunique()
        complete = set(counts.index[counts.eq(month["fecha"].nunique())])
        if preferred[period] in complete:
            continue
        for batch in fallback.get(period, []):
            eligible = complete.intersection(batch)
            if len(eligible) > 1:
                raise IngestionError(
                    f"Revisiones alternativas ambiguas para {period}: {sorted(eligible)}"
                )
            if eligible:
                selected[period] = eligible.pop()
                break
    return selected


def _select_revisions(
    data: pd.DataFrame,
    revisions: dict[str, str] | None,
    fallback: dict[str, list[list[str]]] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    invalid = data["revision"].str.strip().fillna("").eq("")
    if invalid.any():
        hours = data.loc[invalid, "fecha"].drop_duplicates()
        raise IngestionError(
            f"Versiones de liquidación nulas o vacías en {len(hours)} horas; "
            f"ejemplos: {hours.head(3).astype(str).tolist()}"
        )
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
            raise IngestionError(
                f"Revisión oficial ausente en {len(missing)} horas con datos; "
                f"versiones esperadas: {revisions}; ejemplos disponibles: {examples.to_dict('records')}"
            )
        data = selected
    counts = data.groupby("fecha")["revision"].nunique()
    multiple = counts.loc[counts.gt(1)]
    if not multiple.empty:
        examples = data.loc[data["fecha"].isin(multiple.index[:3]), ["fecha", "revision"]]
        examples = examples.drop_duplicates().assign(fecha=lambda rows: rows["fecha"].astype(str))
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
    """Valida las claves naturales antes de construir series a su granularidad nativa."""
    blocks = [_normalize_chunk(chunk, spec, timezone) for chunk in chunks if len(chunk.index)]
    if not blocks:
        raise IngestionError("Fuente sin registros")
    data = pd.concat(blocks, ignore_index=True)
    data = data.loc[data["fecha"].dt.date.between(start, end)]
    if data.empty or not data["valor"].notna().any():
        raise IngestionError("Sin valores en la ventana para los filtros seleccionados")
    selected = {}
    if spec.revision:
        if spec.revision_calendar_id and revisions is None:
            raise IngestionError("Falta el calendario oficial de revisiones de liquidación")
        data, selected = _select_revisions(data, revisions, fallback)

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
    output = (
        data.sort_values(key)[[*key, "valor"]]
        .rename(columns={"fecha": timestamp_column, "valor": value_column})
        .reset_index(drop=True)
    )
    output.attrs["settlement_revisions"] = selected
    return output


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


def _api_chunks(
    session, api, dataset_id, name, windows
) -> Iterator[tuple[str, bytes, dict[str, str]]]:
    endpoint = api["api_base_url"].rstrip("/") + api["endpoint"]
    for start, end in windows:
        params = {"datasetId": dataset_id, "startDate": str(start), "endDate": str(end)}
        response = session.get(endpoint, params=params, timeout=api["timeout_seconds"])
        response.raise_for_status()
        yield f"{name}_{start}_{end}.json", response.content, {"url": response.url, **params}


def _normalized_source(  # noqa: PLR0913
    chunks: list[pd.DataFrame],
    name: str,
    spec: Dataset,
    manifest: dict,
    revisions: dict[str, str] | None,
    *,
    fallback: dict[str, list[list[str]]] | None = None,
):
    settings, window = manifest["settings"], manifest["window"]
    return normalize_dataset(
        chunks,
        spec,
        date.fromisoformat(window["start"]),
        date.fromisoformat(window["end"]),
        settings["timezone"],
        timestamp_column=settings["timestamp_column"],
        value_column=settings["target_column"] if name == "precio_bolsa" else name,
        revisions=revisions,
        fallback=fallback,
    )


def _stage_chunks(
    stage: Path, artifacts: Iterable[tuple[str, bytes, dict[str, str]]], name: str
) -> tuple[list[dict], list[pd.DataFrame]]:
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
    return files, chunks


def _stage_source(stage: Path, artifacts, name: str, spec: Dataset, manifest: dict) -> dict:
    files, chunks = _stage_chunks(stage, artifacts, name)
    calendar = manifest["revision_calendars"].get(name)
    revisions = calendar["revisions"] if calendar is not None else None
    fallback = None
    if calendar is not None and spec.revision_fallback:
        records = pd.concat(_read_snapshot_chunks(stage, calendar["files"]), ignore_index=True)
        fallback = _fallback_revisions(records, spec.revision_as_of)
    normalized = _normalized_source(chunks, name, spec, manifest, revisions, fallback=fallback)
    selected = normalized.attrs["settlement_revisions"]
    if calendar is not None and selected != revisions:
        calendar["preferred_revisions"] = revisions
        calendar["revisions"] = selected
        log_event(
            logging.WARNING,
            "settlement_revision_fallback",
            dataset=name,
            preferred=revisions,
            selected=selected,
        )
    stats = {"normalized_rows": len(normalized)}
    if name == "precio_bolsa":
        settings, window = manifest["settings"], manifest["window"]
        start = pd.Timestamp(window["start"], tz=settings["timezone"])
        stop = pd.Timestamp(window["end"], tz=settings["timezone"]) + pd.DateOffset(days=1)
        grid = pd.date_range(start, stop, freq=settings["frequency"], inclusive="left")
        missing = grid.difference(normalized[settings["timestamp_column"]])
        stats.update(expected_hours=len(grid), missing_hours=len(missing))
        if not missing.empty:
            log_event(
                logging.WARNING,
                "price_hours_missing",
                dataset=name,
                missing_hours=len(missing),
                examples=missing[:3].astype(str).tolist(),
            )
    log_event(
        logging.INFO,
        "dataset_validated",
        dataset=name,
        rows=len(normalized),
        null_values=int(normalized.iloc[:, -1].isna().sum()),
    )
    return {"spec": asdict(spec), "files": files, **stats}


def _stage_revision_calendar(  # noqa: PLR0913
    stage: Path,
    session,
    api: dict,
    spec: Dataset,
    manifest: dict,
    *,
    source_dir: Path | None,
    cached: dict[str, list[dict]],
) -> dict:
    window = manifest["window"]
    start, end = date.fromisoformat(window["start"]), date.fromisoformat(window["end"])
    name = f"calendario_liquidacion_{spec.revision_calendar_id}"
    if name in cached:
        files = cached[name]
        chunks = _read_snapshot_chunks(stage, files)
    else:
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
    revisions = _official_revisions(
        pd.concat(chunks, ignore_index=True), start, end, spec.revision_as_of
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
        "revision_calendars": {},
    }

    with (
        tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as temp,
        _session(api["max_retries"]) as session,
    ):
        stage = Path(temp)
        calendars: dict[str, list[dict]] = {}
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
                if spec.revision_calendar_id and (
                    not spec.revision
                    or not re.fullmatch(r"[a-zA-Z0-9]{6}", spec.revision_calendar_id)
                ):
                    raise IngestionError(
                        "El calendario requiere columna de versión e ID de seis caracteres"
                    )
                if spec.revision_as_of is not None and not spec.revision_calendar_id:
                    raise IngestionError("revision_as_of requiere revision_calendar_id")
                if spec.revision_calendar_id:
                    if spec.revision in spec.filters:
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
                raise IngestionError(f"{name} ({spec.dataset_id}): {exc}") from exc
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(destination)
    log_event(logging.INFO, "snapshot_published", path=destination, start=first, end=last)
    return destination


def _read_snapshot_chunks(snapshot: Path, files: list[dict]) -> list[pd.DataFrame]:
    chunks = []
    for entry in files:
        path = snapshot / entry["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise IngestionError(f"SHA-256 no coincide: {path}")
        chunks.append(records_from_bytes(content, path.suffix))
    return chunks


def read_snapshot(snapshot: Path, name: str) -> pd.DataFrame:
    """Reconstruye una serie con las reglas y el calendario congelados en su manifiesto."""
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise IngestionError("Versión de manifiesto no soportada")
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
    result = _normalized_source(
        _read_snapshot_chunks(snapshot, source["files"]),
        name,
        spec,
        manifest,
        revisions,
        fallback=fallback,
    )
    if spec.revision_calendar_id and result.attrs["settlement_revisions"] != calendar["revisions"]:
        raise IngestionError("Las revisiones reconstruidas no coinciden con el manifiesto")
    return result


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
