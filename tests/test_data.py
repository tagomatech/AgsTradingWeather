import pandas as pd

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
