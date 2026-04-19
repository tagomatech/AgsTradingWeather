import pandas as pd

from agstradingweatherapp.analytics import (
    build_rainfall_threshold_matrix,
    build_recent_context,
    build_snapshot,
    classify_issue,
    describe_comparison_mode,
)
from agstradingweatherapp.config import PALM_OIL


def test_classify_issue_uses_precipitation_labels():
    assert classify_issue(2.0, "precipitation") == "Wet stress"
    assert classify_issue(1.1, "precipitation") == "Wet watch"
    assert classify_issue(-1.1, "precipitation") == "Dry watch"
    assert classify_issue(-2.0, "precipitation") == "Dry stress"


def test_build_rainfall_threshold_matrix_flags_monthly_totals():
    january = pd.date_range("2026-01-01", "2026-01-31", freq="D")
    february = pd.date_range("2026-02-01", "2026-02-28", freq="D")
    records = []
    for date_value in january:
        records.append(
            {
                "date": date_value,
                "date_release": pd.Timestamp("2026-02-28"),
                "year_market": "2025/2026",
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn",
                "geo_label": "Indonesia",
                "geo_level": "country",
                "geo_weight": 100.0,
                "param": "palmoil-tp_sum-mm",
                "value": 18.0,
                "period": "actual",
            }
        )
        records.append(
            {
                "date": date_value,
                "date_release": pd.Timestamp("2026-02-28"),
                "year_market": "2025/2026",
                "country_code": "mys",
                "country_label": "Malaysia",
                "iso_alpha3": "MYS",
                "geo": "mys",
                "geo_label": "Malaysia",
                "geo_level": "country",
                "geo_weight": 100.0,
                "param": "palmoil-tp_sum-mm",
                "value": 2.0,
                "period": "actual",
            }
        )
    for date_value in february:
        records.append(
            {
                "date": date_value,
                "date_release": pd.Timestamp("2026-02-28"),
                "year_market": "2025/2026",
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn",
                "geo_label": "Indonesia",
                "geo_level": "country",
                "geo_weight": 100.0,
                "param": "palmoil-tp_sum-mm",
                "value": 10.0,
                "period": "actual",
            }
        )

    country_daily = pd.DataFrame.from_records(records)
    matrix, scope_label = build_rainfall_threshold_matrix(
        country_daily=country_daily,
        geo_daily=pd.DataFrame(),
        crop=PALM_OIL,
        scope="all",
        months=2,
    )

    january_idn = matrix[(matrix["row_label"] == "Indonesia") & (matrix["month_label"] == "Jan\n2026")].iloc[0]
    january_mys = matrix[(matrix["row_label"] == "Malaysia") & (matrix["month_label"] == "Jan\n2026")].iloc[0]
    february_idn = matrix[(matrix["row_label"] == "Indonesia") & (matrix["month_label"] == "Feb\n2026")].iloc[0]

    assert scope_label == "Whole countries"
    assert january_idn["bucket"] == 1
    assert january_mys["bucket"] == -1
    assert february_idn["bucket"] == 0


def test_build_recent_context_uses_last_year_reference_by_default():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-04-01"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 25.0,
            },
            {
                "date": pd.Timestamp("2025-04-02"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 26.0,
            },
            {
                "date": pd.Timestamp("2025-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 27.0,
            },
            {
                "date": pd.Timestamp("2026-04-01"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 28.0,
            },
            {
                "date": pd.Timestamp("2026-04-02"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 29.0,
            },
            {
                "date": pd.Timestamp("2026-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 30.0,
            },
        ]
    )

    recent_context, scope_label = build_recent_context(
        country_daily=frame,
        geo_daily=pd.DataFrame(),
        crop=PALM_OIL,
        scope="idn",
        current_date=pd.Timestamp("2026-04-03"),
        lookback_days=5,
        forward_days=0,
        window_days=3,
        comparison_mode="same_period_last_year",
    )

    current_cut = recent_context[recent_context["date"] == pd.Timestamp("2026-04-03")].iloc[0]
    assert scope_label == "Indonesia"
    assert current_cut["reference_mean"] == 27.0
    assert current_cut["reference_low"] == 27.0
    assert current_cut["reference_high"] == 27.0


def test_build_recent_context_can_switch_to_previous_window_reference():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-29"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 24.0,
            },
            {
                "date": pd.Timestamp("2026-03-30"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 25.0,
            },
            {
                "date": pd.Timestamp("2026-03-31"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 26.0,
            },
            {
                "date": pd.Timestamp("2026-04-01"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 27.0,
            },
            {
                "date": pd.Timestamp("2026-04-02"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 28.0,
            },
            {
                "date": pd.Timestamp("2026-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 29.0,
            },
            {
                "date": pd.Timestamp("2026-04-04"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 30.0,
            },
        ]
    )

    recent_context, _ = build_recent_context(
        country_daily=frame,
        geo_daily=pd.DataFrame(),
        crop=PALM_OIL,
        scope="idn",
        current_date=pd.Timestamp("2026-04-04"),
        lookback_days=6,
        forward_days=0,
        window_days=3,
        comparison_mode="previous_window",
    )

    current_cut = recent_context[recent_context["date"] == pd.Timestamp("2026-04-04")].iloc[0]
    assert current_cut["reference_mean"] == 28.0
    assert pd.isna(current_cut["reference_low"])
    assert pd.isna(current_cut["reference_high"])


def test_describe_comparison_mode_supports_historical_normal():
    assert describe_comparison_mode("seasonal_normal", 14) == "historical normal"


def test_build_snapshot_uses_window_totals_for_accumulated_precipitation():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-04-01"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 1.0,
            },
            {
                "date": pd.Timestamp("2025-04-02"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 2.0,
            },
            {
                "date": pd.Timestamp("2025-04-03"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 3.0,
            },
            {
                "date": pd.Timestamp("2026-04-01"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 4.0,
            },
            {
                "date": pd.Timestamp("2026-04-02"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 5.0,
            },
            {
                "date": pd.Timestamp("2026-04-03"),
                "country_code": "idn",
                "country_label": "Indonesia",
                "iso_alpha3": "IDN",
                "geo": "idn-riau",
                "geo_label": "Riau",
                "geo_level": "region",
                "geo_weight": 22.1,
                "param": "palmoil-tp_sum-mm",
                "value": 6.0,
            },
        ]
    )

    snapshot = build_snapshot(
        frame,
        current_date=pd.Timestamp("2026-04-03"),
        window_days=3,
        comparison_mode="same_period_last_year",
    )

    current_cut = snapshot.iloc[0]
    assert current_cut["window_mean"] == 15.0
    assert current_cut["reference_mean"] == 6.0
    assert current_cut["anomaly"] == 9.0


def test_build_recent_context_historical_normal_uses_all_prior_years():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 24.0,
            },
            {
                "date": pd.Timestamp("2025-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 28.0,
            },
            {
                "date": pd.Timestamp("2026-04-03"),
                "country_code": "idn",
                "param": "palmoil-t2m_mean-degree_c",
                "value": 30.0,
            },
        ]
    )

    recent_context, _ = build_recent_context(
        country_daily=frame,
        geo_daily=pd.DataFrame(),
        crop=PALM_OIL,
        scope="idn",
        current_date=pd.Timestamp("2026-04-03"),
        lookback_days=1,
        forward_days=0,
        window_days=3,
        comparison_mode="seasonal_normal",
    )

    current_cut = recent_context.iloc[0]
    assert current_cut["reference_mean"] == 26.0
    assert current_cut["reference_low"] == 24.4
    assert current_cut["reference_high"] == 27.6
