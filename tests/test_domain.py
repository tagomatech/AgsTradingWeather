import pandas as pd

from agstradingweatherapp.config import PALM_OIL
from agstradingweatherapp.data import enrich_geo_daily, prepare_weather_frame, reduce_to_country_level
from agstradingweatherapp.domain import parse_param_label


def test_parse_param_label_extracts_expected_tokens():
    descriptor = parse_param_label("palmoil-t2m_max-degree_c")

    assert descriptor.crop_key == "palmoil"
    assert descriptor.variable_code == "t2m"
    assert descriptor.statistic_code == "max"
    assert descriptor.unit_code == "degree_c"
    assert descriptor.short_label == "Maximum Temperature"


def test_reduce_to_country_level_prefers_exact_country_geo():
    raw = pd.DataFrame(
        [
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "idn",
                "value": 28.4,
            },
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "idn-riau",
                "value": 27.1,
            },
        ]
    )

    prepared = prepare_weather_frame(raw)
    geo_daily = enrich_geo_daily(
        prepared=prepared,
        crop=PALM_OIL,
        current_date=pd.Timestamp("2026-04-13"),
    )
    reduced = reduce_to_country_level(
        geo_daily=geo_daily,
        crop=PALM_OIL,
        current_date=pd.Timestamp("2026-04-13"),
    )

    assert len(reduced) == 1
    assert reduced.iloc[0]["country_code"] == "idn"
    assert reduced.iloc[0]["value"] == 28.4


def test_reduce_to_country_level_uses_region_weights_when_country_missing():
    raw = pd.DataFrame(
        [
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "mys-sabah",
                "value": 27.0,
            },
            {
                "date": "2026-04-13",
                "date_release": "2026-04-13",
                "year_market": 2026,
                "param": "palmoil-t2m_mean-degree_c",
                "geo": "mys-sarawak",
                "value": 26.0,
            },
        ]
    )

    prepared = prepare_weather_frame(raw)
    geo_daily = enrich_geo_daily(
        prepared=prepared,
        crop=PALM_OIL,
        current_date=pd.Timestamp("2026-04-13"),
    )
    reduced = reduce_to_country_level(
        geo_daily=geo_daily,
        crop=PALM_OIL,
        current_date=pd.Timestamp("2026-04-13"),
    )

    assert len(reduced) == 1
    assert reduced.iloc[0]["geo"] == "mys"
    assert reduced.iloc[0]["value"] > 26.5
