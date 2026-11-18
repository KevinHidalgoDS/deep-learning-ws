import pandas as pd
import pytest
from src.data.price_eda import (
    XM_HOURS,
    audit_dataframe,
    compute_extended_statistics,
    melt_hourly_to_long,
)


def test_xm_wide_to_long_preserves_local_hours_and_missing_values() -> None:
    row = dict(zip(XM_HOURS, map(str, range(1, 25)), strict=True))
    wide = pd.DataFrame([{**row, "Date": "2025-01-02", "Values_Hour12": None}])
    summary = audit_dataframe(wide).set_index("Columna")
    assert summary.loc["Values_Hour12", "Pct_Nulos"] == 100.0  # noqa: PLR2004

    long = melt_hourly_to_long(wide, "Precio")
    expected = pd.date_range("2025-01-02", periods=24, freq="h", tz="America/Bogota")
    assert pd.DatetimeIndex(long["Timestamp"]).equals(expected)
    assert long["Hour_Int"].tolist() == list(range(1, 25))
    assert long["Precio"].dtype == "float64"
    assert pd.isna(long["Precio"].iloc[11])


def test_xm_rejects_missing_hour_columns_and_duplicate_dates() -> None:
    wide = pd.DataFrame([{**dict.fromkeys(XM_HOURS, 1), "Date": "2025-01-01"}])
    with pytest.raises(ValueError, match="Values_Hour24"):
        melt_hourly_to_long(wide.drop(columns="Values_Hour24"), "Precio")
    with pytest.raises(ValueError, match="misma fecha"):
        melt_hourly_to_long(pd.concat([wide, wide]), "Precio")


def test_extended_statistics_counts_tukey_outliers_and_handles_missing_data() -> None:
    stats = compute_extended_statistics(pd.Series([1, 1, 2, 2, 100, None]))
    assert (stats["Observaciones"], stats["Outliers_IQR_Count"], stats["Outliers_IQR_%"]) == (
        5,
        1,
        20,
    )
    empty = compute_extended_statistics(pd.Series([float("nan")]))
    assert empty["Observaciones"] == 0
    assert pd.isna(empty["Outliers_IQR_%"])
