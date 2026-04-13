import pandas as pd
import pytest

from agstradingapp.config import PALM_OIL
from agstradingapp.data import load_dataset


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

    monkeypatch.setenv("AGSTRADINGAPP_DATA_FILE", str(csv_path))

    dataset = load_dataset(PALM_OIL)

    assert dataset.source_mode == "csv"
    assert not dataset.geo_daily.empty
    assert not dataset.country_daily.empty
    assert dataset.current_date == pd.Timestamp("2026-04-12")


def test_load_dataset_raises_when_csv_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGSTRADINGAPP_DATA_FILE", raising=False)

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

    monkeypatch.delenv("AGSTRADINGAPP_DATA_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    dataset = load_dataset(PALM_OIL)

    assert dataset.source_mode == "csv"
    assert "palm_oil_weather_feed.csv" in dataset.status_message
    assert dataset.current_date == pd.Timestamp("2026-04-13")
