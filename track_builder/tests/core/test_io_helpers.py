import pandas as pd
import numpy as np
from pathlib import Path

from track_builder.core.io_helpers import (
    iter_files,
    read_csv_auto,
    standardize_columns,
    parse_dates,
    validate_coords,
    quality_filter,
    matches_year_month,
    DEFAULT_DATA_PATH,
)


# ---------- iter_files ----------

def test_iter_files_single_file(tmp_path: Path):
    f = tmp_path / "file.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    files = iter_files(f, pattern=None)
    assert files == [f.resolve()]


def test_iter_files_list_of_files(tmp_path: Path):
    f1 = tmp_path / "a.csv"
    f1.write_text("x\n", encoding="utf-8")
    f2 = tmp_path / "b.csv"
    f2.write_text("y\n", encoding="utf-8")
    files = iter_files([f1, f2], pattern=None)
    assert set(files) == {f1.resolve(), f2.resolve()}


def test_iter_files_directory_with_pattern(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    f1 = d / "ASTD_201907.csv"
    f1.write_text("x\n", encoding="utf-8")
    f2 = d / "ASTD_201908.csv"
    f2.write_text("y\n", encoding="utf-8")
    files = iter_files(d, pattern="ASTD_*.csv")
    assert set(p.name for p in files) == {"ASTD_201907.csv", "ASTD_201908.csv"}


def test_iter_files_none_uses_default_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("track_builder.core.io_helpers.DEFAULT_DATA_PATH", tmp_path)
    f = tmp_path / "x.csv"
    f.write_text("z\n", encoding="utf-8")
    files = iter_files(None, pattern=None)
    assert files == [f.resolve()]


# ---------- read_csv_auto ----------

def test_read_csv_auto_detects_comma(tmp_path: Path):
    f = tmp_path / "c.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    df = read_csv_auto(f)
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0, 0] == 1


def test_read_csv_auto_detects_semicolon(tmp_path: Path):
    f = tmp_path / "s.csv"
    f.write_text("a;b\n1;2\n", encoding="utf-8")
    df = read_csv_auto(f)
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0, 1] == 2


# ---------- standardize_columns ----------

def test_standardize_columns_basic():
    df = pd.DataFrame({
        "LAT": [60.0],
        "Lon": [-20.0],
        "Datetime_UTC": ["2019-07-01T00:00:00Z"],
        "FlagName": [" Panama "],
        "ASTD_Cat": ["Container Ships"],
        "shipid": ["s1"],
    })
    d2 = standardize_columns(df)
    assert {"latitude", "longitude", "date_time_utc", "flagname", "astd_cat", "shipid"} <= set(d2.columns)


# ---------- parse_dates ----------

def test_parse_dates_to_utc():
    df = pd.DataFrame({
        "date_time_utc": ["2019-07-01T00:00:00Z", "2019-07-01T01:00:00Z"]
    })
    d2 = parse_dates(df)
    assert pd.api.types.is_datetime64tz_dtype(d2["date_time_utc"])
    assert str(d2["date_time_utc"].dt.tz) == "UTC"


def test_parse_dates_from_alias():
    df = pd.DataFrame({"Datetime_UTC": ["2019-07-01 00:00:00+00:00"]})
    d2 = standardize_columns(df)
    d2 = parse_dates(d2)
    assert "date_time_utc" in d2.columns
    assert pd.api.types.is_datetime64tz_dtype(d2["date_time_utc"])


# ---------- validate_coords ----------

def test_validate_coords_filters_invalids():
    df = pd.DataFrame({
        "latitude": [60, 95, np.nan],
        "longitude": [-20, 181, -10],
    })
    d2 = validate_coords(df)
    # kept only (60, -20)
    assert len(d2) == 1
    assert d2.iloc[0]["latitude"] == 60
    assert d2.iloc[0]["longitude"] == -20


# ---------- quality_filter ----------

def test_quality_filter_threshold_minutes():
    t_good = pd.to_datetime(
        ["2019-07-01T00:00:00Z", "2019-07-01T00:10:00Z", "2019-07-01T00:20:00Z"], utc=True
    )
    t_bad = pd.to_datetime(
        ["2019-07-02T00:00:00Z", "2019-07-02T03:00:00Z", "2019-07-02T06:00:00Z"], utc=True
    )
    df = pd.DataFrame({
        "shipid": ["A"] * 3 + ["B"] * 3,
        "date_time_utc": list(t_good) + list(t_bad),
        "latitude": [60] * 6, "longitude": [-20] * 6,
    })
    d2 = quality_filter(df, threshold_minutes=60)
    assert set(d2["shipid"]) == {"A"}


def test_quality_filter_noop_when_threshold_zero():
    df = pd.DataFrame({
        "shipid": ["X", "X"],
        "date_time_utc": pd.to_datetime(["2019-07-01T00:00:00Z", "2019-07-01T01:00:00Z"], utc=True),
        "latitude": [0, 0], "longitude": [0, 0],
    })
    d2 = quality_filter(df, threshold_minutes=0)
    assert len(d2) == 2  # pas de filtrage


# ---------- matches_year_month ----------

def test_matches_year_month_true_variants():
    assert matches_year_month("ASTD_area_level3_201907.csv", 2019, {7})
    assert matches_year_month("x_2019-08_y.csv", 2019, {8})
    assert matches_year_month("x_2019_09.csv", 2019, {9})


def test_matches_year_month_false_cases():
    assert not matches_year_month("ASTD_201906.csv", 2019, {7, 8, 9})
    assert not matches_year_month("ASTD_2019.csv", 2019, {7, 8, 9})
