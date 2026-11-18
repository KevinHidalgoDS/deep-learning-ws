import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from src.config import read_config_mapping
from src.data.make_dataset import (
    Dataset,
    IngestionError,
    date_chunks,
    ingest,
    normalize_dataset,
    read_snapshot,
    records_from_bytes,
)

DAY = date(2024, 4, 1)
TZ = "America/Bogota"
BASE_ROW = {
    "FechaHora": "2024-04-01 01:00:00",
    "Valor": "123.50",
    "CodigoVariable": "PB_Nal",
    "Version": "TX5",
    "CodigoDuracion": "PT1H",
    "UnidadMedida": "COP/kWh",
}


@pytest.fixture
def spec() -> Dataset:
    settings = read_config_mapping()["simem"]["datasets"]["precio_bolsa"]
    return replace(
        Dataset(**settings),
        revision=None,
        revision_calendar_id=None,
        filters={**settings["filters"], "Version": "TX5"},
    )


def test_price_schema_types_and_unique_hours(spec: Dataset) -> None:
    raw = pd.DataFrame(
        [
            BASE_ROW,
            {**BASE_ROW, "Version": "TX1", "Valor": "99"},
            {**BASE_ROW, "CodigoVariable": "PB_Int", "Valor": "900"},
            {**BASE_ROW, "FechaHora": "2024-04-01 00:00:00", "Valor": None},
        ]
    )
    price = normalize_dataset([raw], spec, DAY, DAY, TZ)

    assert price.columns.tolist() == ["fecha", "precio_bolsa"]
    assert isinstance(price["fecha"].dtype, pd.DatetimeTZDtype)
    assert str(price["fecha"].dt.tz) == TZ
    assert price["precio_bolsa"].dtype == "float64"
    assert price["fecha"].is_unique and price["fecha"].is_monotonic_increasing
    assert price["fecha"].dt.hour.tolist() == [0, 1]
    assert pd.isna(price["precio_bolsa"].iloc[0])
    assert price["precio_bolsa"].iloc[1] == float(BASE_ROW["Valor"])


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("Valor", "no-numérico", "Unable to parse"),
        ("Valor", "inf", "infinitos"),
        ("FechaHora", "2024-04-01 01:45:00", "fuera de la malla"),
        ("FechaHora", None, "Fechas nulas"),
    ],
)
def test_rejects_bad_types_and_off_grid_timestamps(
    spec: Dataset, column: str, bad_value: str | None, message: str
) -> None:
    raw = pd.DataFrame([{**BASE_ROW, column: bad_value}])
    with pytest.raises(ValueError, match=message):
        normalize_dataset([raw], spec, DAY, DAY, TZ)


def test_rejects_missing_required_column(spec: Dataset) -> None:
    raw = pd.DataFrame([BASE_ROW]).drop(columns="Valor")
    with pytest.raises(IngestionError, match=r"faltantes.*Valor"):
        normalize_dataset([raw], spec, DAY, DAY, TZ)


def test_rejects_duplicate_hours_across_chunks(spec: Dataset) -> None:
    raw = pd.DataFrame([BASE_ROW])
    equivalent_utc = raw.assign(FechaHora="2024-04-01T06:00:00Z")
    with pytest.raises(IngestionError, match="Duplicados"):
        normalize_dataset([raw, equivalent_utc], spec, DAY, DAY, TZ)


def test_checks_plant_key_before_daily_aggregation() -> None:
    spec = Dataset(**read_config_mapping()["simem"]["datasets"]["generacion_real"])
    row = {
        "Fecha": str(DAY),
        "TipoGeneracion": "Hidraulica",
        "CodigoPlanta": "AAA",
        "GeneracionRealEstimada": 10,
    }
    raw = pd.DataFrame(
        [
            row,
            {**row, "CodigoPlanta": "BBB", "GeneracionRealEstimada": 15},
            {
                **row,
                "TipoGeneracion": "Termica",
                "CodigoPlanta": "CCC",
                "GeneracionRealEstimada": 9,
            },
            {**row, "TipoGeneracion": "Solar"},
        ]
    )
    data = normalize_dataset([raw], spec, DAY, DAY, TZ, value_column="generacion_real")
    assert data.columns.tolist() == ["fecha", "TipoGeneracion", "generacion_real"]
    assert not data.duplicated(["fecha", "TipoGeneracion"]).any()
    assert data["generacion_real"].tolist() == [25.0, 9.0]
    with pytest.raises(IngestionError, match="Duplicados"):
        normalize_dataset([raw, raw.iloc[:1]], spec, DAY, DAY, TZ)


def test_demand_does_not_mix_revisions_within_each_hour() -> None:
    spec = replace(
        Dataset(**read_config_mapping()["simem"]["datasets"]["demanda_nacional"]),
        revision_calendar_id=None,
    )
    row = {
        **BASE_ROW,
        "CodigoVariable": "DdaReal",
        "UnidadMedida": "kWh",
        "CodigoSICAgente": "AAA",
        "TipoMercado": "Regulado",
        "Version": "TX6",
    }
    disjoint = {**row, "FechaHora": "2024-04-01 02:00:00", "Version": "TX5"}
    demand = normalize_dataset(
        [pd.DataFrame([row, disjoint])], spec, DAY, DAY, TZ, value_column="demanda_nacional"
    )
    assert demand["fecha"].is_unique
    assert demand["demanda_nacional"].tolist() == [123.5, 123.5]
    overlapping = {**disjoint, "FechaHora": row["FechaHora"], "CodigoSICAgente": "BBB"}
    with pytest.raises(IngestionError, match="múltiples"):
        normalize_dataset([pd.DataFrame([row, overlapping])], spec, DAY, DAY, TZ)


@pytest.mark.parametrize(
    ("as_of", "version", "expected", "demand_has_tx5"),
    [(None, "TX5", 200, True), ("2024-06-30", "TX4", 100, True), (None, "TX5", 200, False)],
)
def test_snapshot_selects_official_price_and_demand_revisions(
    tmp_path: Path,
    as_of: str | None,
    version: str,
    expected: int,
    demand_has_tx5: bool,
) -> None:
    config = read_config_mapping()
    sources = config["simem"]["datasets"]
    config["simem"]["datasets"] = {
        name: sources[name] for name in ("precio_bolsa", "demanda_nacional")
    }
    sources["demanda_nacional"]["revision_as_of"] = as_of
    sources["precio_bolsa"]["revision_as_of"] = as_of
    pd.DataFrame(
        [
            {**BASE_ROW, "Version": "TX4", "Valor": 100},
            {**BASE_ROW, "Version": "TX5", "Valor": 200},
        ]
    ).to_csv(tmp_path / "precio_bolsa.csv", index=False)
    demand = {
        **BASE_ROW,
        "CodigoVariable": "DdaReal",
        "UnidadMedida": "kWh",
        "CodigoSICAgente": "AAA",
        "TipoMercado": "Regulado",
    }
    demand_rows = [{**demand, "Version": "TX4", "Valor": 100}]
    if demand_has_tx5:
        demand_rows.append({**demand, "Version": "TX5", "Valor": 200})
    pd.DataFrame(demand_rows).to_csv(tmp_path / "demanda_nacional.csv", index=False)
    period = {"FechaInicio": "2024-04-01", "FechaFin": "2024-04-30"}
    pd.DataFrame(
        [
            {**period, "Version": "TX4", "FechaPublicacion": "2024-06-30", "EsMaximaVersion": 0},
            {**period, "Version": "TX5", "FechaPublicacion": "2024-07-01", "EsMaximaVersion": 1},
        ]
    ).to_csv(tmp_path / "calendario_liquidacion_24914F.csv", index=False)

    snapshot = ingest(config, source_dir=tmp_path, root=tmp_path, start=str(DAY), end=str(DAY))
    result = read_snapshot(snapshot, "demanda_nacional")
    assert result["fecha"].is_unique
    assert result["demanda_nacional"].tolist() == ([expected] if demand_has_tx5 else [100])
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    price = read_snapshot(snapshot, "precio_bolsa")
    assert price["fecha"].is_unique and price["precio_bolsa"].tolist() == [expected]
    expected_hours_in_day = 24
    assert manifest["datasets"]["precio_bolsa"]["missing_hours"] == expected_hours_in_day - len(
        price
    )
    calendars = manifest["revision_calendars"]
    assert calendars["precio_bolsa"]["revisions"] == {"2024-04": version}
    assert calendars["demanda_nacional"]["revisions"] == {
        "2024-04": version if demand_has_tx5 else "TX4"
    }
    assert calendars["precio_bolsa"]["files"] == calendars["demanda_nacional"]["files"]
    if not demand_has_tx5:
        assert calendars["demanda_nacional"]["preferred_revisions"] == {"2024-04": "TX5"}
    (snapshot / "calendario_liquidacion_24914F.csv").write_bytes(b"alterado")
    with pytest.raises(IngestionError, match="SHA-256"):
        read_snapshot(snapshot, "demanda_nacional")


@pytest.mark.parametrize(
    ("provided", "message"), [(None, "nulas o vacías"), ("TX4", "oficial ausente")]
)
def test_demand_diagnoses_null_or_missing_official_revision(
    provided: str | None, message: str
) -> None:
    spec = Dataset(**read_config_mapping()["simem"]["datasets"]["demanda_nacional"])
    row = {
        **BASE_ROW,
        "CodigoVariable": "DdaReal",
        "UnidadMedida": "kWh",
        "CodigoSICAgente": "AAA",
        "TipoMercado": "Regulado",
        "Version": provided,
    }
    with pytest.raises(IngestionError, match=message):
        normalize_dataset([pd.DataFrame([row])], spec, DAY, DAY, TZ, revisions={"2024-04": "TX5"})


def test_request_windows_do_not_overlap() -> None:
    assert list(date_chunks(DAY, date(2024, 4, 5), 2)) == [
        (DAY, date(2024, 4, 2)),
        (date(2024, 4, 3), date(2024, 4, 4)),
        (date(2024, 4, 5), date(2024, 4, 5)),
    ]


def test_public_data_response_contract() -> None:
    payload = json.dumps({"success": True, "result": {"records": [BASE_ROW]}}).encode()
    assert records_from_bytes(payload, ".json").iloc[0].to_dict() == BASE_ROW
    with pytest.raises(IngestionError, match="success=true"):
        records_from_bytes(b'{"success": false}', ".json")


@pytest.fixture
def local_ingestion(tmp_path: Path, spec: Dataset) -> tuple[dict, Path, Path]:
    config = read_config_mapping()
    config["simem"]["datasets"]["precio_bolsa"].update(
        filters=spec.filters,
        revision=None,
        revision_calendar_id=None,
    )
    config["simem"]["datasets"] = {"precio_bolsa": config["simem"]["datasets"]["precio_bolsa"]}
    source = tmp_path / "exports"
    source.mkdir()
    path = source / "precio_bolsa.csv"
    pd.DataFrame([BASE_ROW]).to_csv(path, index=False)
    return config, source, path


def test_local_snapshots_preserve_original_bytes_and_do_not_overwrite(
    tmp_path: Path, local_ingestion: tuple[dict, Path, Path]
) -> None:
    config, source, export = local_ingestion
    args = {"source_dir": source, "root": tmp_path, "start": str(DAY), "end": str(DAY)}
    original = export.read_bytes()
    first = ingest(config, **args)
    pd.DataFrame([{**BASE_ROW, "Valor": "321.50"}]).to_csv(export, index=False)
    second = ingest(config, **args)

    assert first != second
    assert (first / export.name).read_bytes() == original
    assert (second / export.name).read_bytes() == export.read_bytes()
    assert read_snapshot(first, "precio_bolsa")["precio_bolsa"].tolist() == [123.5]
    assert read_snapshot(second, "precio_bolsa")["precio_bolsa"].tolist() == [321.5]
    (first / export.name).write_bytes(b"alterado")
    with pytest.raises(IngestionError, match="SHA-256"):
        read_snapshot(first, "precio_bolsa")


def test_failed_ingestion_does_not_publish_snapshot(
    tmp_path: Path, local_ingestion: tuple[dict, Path, Path]
) -> None:
    config, source, export = local_ingestion
    pd.DataFrame([BASE_ROW, BASE_ROW]).to_csv(export, index=False)
    with pytest.raises(IngestionError, match="Duplicados"):
        ingest(config, source_dir=source, root=tmp_path, start=str(DAY), end=str(DAY))
    assert not list((tmp_path / "data" / "raw").rglob("manifest.json"))
    assert not list((tmp_path / "data" / "raw").rglob(".staging-*"))
