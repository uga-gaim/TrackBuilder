from pathlib import Path
import pandas as pd

from track_builder.io.astd_loader import load_astd_data, load_astd_monthly


def _write_csv(path: Path, sep: str = ",", month: str = "2019-07"):
    if sep == ",":
        txt = (
            "shipid,date_time_utc,latitude,longitude,astd_cat,flagname,iceclass,sizegroup_gt\n"
            f"s1,{month}-15T10:00:00Z,60.0,-20.0,container ships,panama,,10000-24999 gt\n"
        )
    else:
        txt = (
            "shipid;date_time_utc;latitude;longitude;astd_cat;flagname;iceclass;sizegroup_gt\n"
            f"s1;{month}-15T10:00:00Z;61.0;-21.0;container ships;panama;;10000-24999 gt\n"
        )
    path.write_text(txt, encoding="utf-8")


# ---------- load_astd_data ----------

def test_load_astd_data_single_file(tmp_path: Path):
    f = tmp_path / "ASTD_area_level3_201907.csv"
    _write_csv(f, sep=",", month="2019-07")
    df = load_astd_data(f, progress=False)
    assert len(df) == 1
    assert {"shipid", "date_time_utc", "latitude", "longitude", "month"} <= set(df.columns)
    assert df.loc[0, "month"] == "2019-07"


def test_load_astd_data_multiple_files_auto_merge(tmp_path: Path):
    f1 = tmp_path / "ASTD_area_level3_201907.csv"
    f2 = tmp_path / "ASTD_area_level3_201908.csv"
    _write_csv(f1, sep=",", month="2019-07")
    _write_csv(f2, sep=";", month="2019-08")  # test séparateur
    df = load_astd_data([f1, f2], progress=False)
    assert len(df) == 2
    assert set(df["month"]) == {"2019-07", "2019-08"}


def test_load_astd_data_directory_with_pattern(tmp_path: Path):
    d = tmp_path / "2019"
    d.mkdir()
    f1 = d / "ASTD_area_level3_201907.csv"
    _write_csv(f1, month="2019-07")
    f2 = d / "ASTD_area_level3_201908.csv"
    _write_csv(f2, month="2019-08")
    df = load_astd_data(d, pattern="ASTD_*.csv", progress=False)
    assert len(df) == 2


# ---------- load_astd_monthly ----------

def test_load_astd_monthly_specific_months(tmp_path: Path):
    f1 = tmp_path / "ASTD_area_level3_201907.csv"
    _write_csv(f1, month="2019-07")
    f3 = tmp_path / "ASTD_area_level3_201909.csv"
    _write_csv(f3, month="2019-09")
    df = load_astd_monthly(tmp_path, 2019, months=[7, 9], progress=False)
    assert set(df["month"]) == {"2019-07", "2019-09"}


def test_load_astd_monthly_full_year(tmp_path: Path):
    # create 12 files
    for m in range(1, 13):
        f = tmp_path / f"ASTD_area_level3_2019{m:02d}.csv"
        _write_csv(f, month=f"2019-{m:02d}")
    df = load_astd_monthly(tmp_path, 2019, progress=False)
    assert set(df["month"]) == {f"2019-{m:02d}" for m in range(1, 13)}


def test_load_astd_monthly_with_pattern_prefilter(tmp_path: Path):
    f1 = tmp_path / "ASTD_area_level3_201907.csv"
    _write_csv(f1, month="2019-07")
    f2 = tmp_path / "ASTD_area_level3_201908.csv"
    _write_csv(f2, month="2019-08")
    f_old = tmp_path / "ASTD_area_level3_201806.csv"
    _write_csv(f_old, month="2018-06")

    df = load_astd_monthly(tmp_path, 2019, months=[7, 8], pattern="ASTD_*.csv", progress=False)
    assert set(df["month"]) == {"2019-07", "2019-08"}
    assert "2018-06" not in set(df["month"])
