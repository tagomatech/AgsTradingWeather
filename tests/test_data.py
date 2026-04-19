import pandas as pd
import pytest

from agstradingweatherapp.config import PALM_OIL
from agstradingweatherapp import data as data_module
from agstradingweatherapp.data import build_core_belt_daily, load_dataset


def test_load_dataset_prefers_local_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / PALM_OIL.data_filename
    pd.DataFrame(
        [
            {
                "date": "2026-04-12",
                "date_release": "2026-04-12",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "idn",
                "value": 28.1,
            },
            {
                "date": "2026-04-12",
                "date_release": "2026-04-12",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "idn-riau",
                "value": 28.4,
            },
            {
                "date": "2026-04-12",
                "date_release": "2026-04-12",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "mys",
                "value": 27.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setenv("AGSTRADINGWEATHERAPP_DATA_FILE", str(csv_path))

    dataset = load_dataset(PALM_OIL)

    assert dataset.source_mode == "csv"
    assert not dataset.geo_daily.empty
    assert not dataset.country_daily.empty
    assert dataset.current_date == pd.Timestamp("2026-04-12")


def test_load_dataset_raises_when_csv_is_missing(monkeypatch, tmp_path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(data_module, "repo_root", lambda: fake_repo)
    monkeypatch.delenv("AGSTRADINGWEATHERAPP_DATA_FILE", raising=False)

    with pytest.raises(FileNotFoundError, match="Palm oil CSV feed not found"):
        load_dataset(PALM_OIL)


def test_load_dataset_finds_csv_in_current_working_directory(monkeypatch, tmp_path):
    csv_path = tmp_path / PALM_OIL.data_filename
    pd.DataFrame(
        [
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_min-degree_c",
                "geo": "mys-sabah",
                "value": 23.9,
            },
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_min-degree_c",
                "geo": "mys",
                "value": 23.7,
            },
        ]
    ).to_csv(csv_path, index=False)

    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(data_module, "repo_root", lambda: fake_repo)
    monkeypatch.delenv("AGSTRADINGWEATHERAPP_DATA_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    dataset = load_dataset(PALM_OIL)

    assert dataset.source_mode == "csv"
    assert "palm_oil_weather_feed.csv" in dataset.status_message
    assert dataset.current_date == pd.Timestamp("2026-04-13")


def test_build_core_belt_daily_uses_country_weights():
    country_daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-04-13"),
                "date_release": pd.Timestamp("2026-04-13"),
                "param": "palmoil-t2m_mean-degree_c",
                "country_code": "idn",
                "value": 28.0,
            },
            {
                "date": pd.Timestamp("2026-04-13"),
                "date_release": pd.Timestamp("2026-04-13"),
                "param": "palmoil-t2m_mean-degree_c",
                "country_code": "mys",
                "value": 26.0,
            },
        ]
    )

    core_belt = build_core_belt_daily(
        country_daily=country_daily,
        crop=PALM_OIL,
        current_date=pd.Timestamp("2026-04-13"),
    )

    assert len(core_belt) == 1
    assert core_belt.iloc[0]["value"] == pytest.approx((28.0 * 0.62) + (26.0 * 0.38))
