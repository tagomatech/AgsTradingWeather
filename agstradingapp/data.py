from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CropDefinition, PALM_OIL
from .domain import build_param_dictionary


@dataclass(frozen=True)
class CropDataset:
    crop: CropDefinition
    raw: pd.DataFrame
    country_daily: pd.DataFrame
    param_dictionary: pd.DataFrame
    current_date: pd.Timestamp
    source_mode: str
    status_message: str


def load_dataset(crop: CropDefinition = PALM_OIL) -> CropDataset:
    db_url = os.getenv("WEATHER_DB_URL")
    status_message = "Using deterministic sample data. Set WEATHER_DB_URL to query source.wth."
    source_mode = "sample"

    if db_url:
        try:
            raw = fetch_live_weather(crop=crop, db_url=db_url)
            source_mode = "database"
            status_message = "Live query succeeded against source.wth."
        except Exception as exc:  # pragma: no cover - exercised only with live DB issues
            raw = generate_sample_weather(crop)
            status_message = (
                "Live query failed, so the app fell back to deterministic sample data. "
                f"Reason: {exc}"
            )
    else:
        raw = generate_sample_weather(crop)

    prepared = prepare_weather_frame(raw)
    current_date = prepared["date_release"].max().normalize()
    country_daily = reduce_to_country_level(prepared, crop=crop, current_date=current_date)
    param_dictionary = build_param_dictionary(country_daily["param"].unique())

    return CropDataset(
        crop=crop,
        raw=prepared,
        country_daily=country_daily,
        param_dictionary=param_dictionary,
        current_date=current_date,
        source_mode=source_mode,
        status_message=status_message,
    )


def fetch_live_weather(crop: CropDefinition, db_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    table_name = os.getenv("WEATHER_SQL_TABLE", "source.wth")
    start_date = os.getenv("WEATHER_START_DATE", "2005-01-01")

    param_clauses: list[str] = []
    geo_clauses: list[str] = []
    bind_params: dict[str, str] = {"start_date": start_date}

    for idx, param in enumerate(crop.params):
        key = f"param_{idx}"
        bind_params[key] = param
        param_clauses.append(f"param = :{key}")

    for idx, country_code in enumerate(crop.country_codes):
        key = f"geo_{idx}"
        bind_params[key] = f"{country_code}%"
        geo_clauses.append(f"geo LIKE :{key}")

    query = f"""
        SELECT date, date_release, year_market, param, geo, value
        FROM {table_name}
        WHERE ({' OR '.join(param_clauses)})
          AND ({' OR '.join(geo_clauses)})
          AND date >= :start_date
          AND geo NOT LIKE '%-%'
        ORDER BY date ASC, date_release ASC, param, geo
    """

    engine = create_engine(db_url)
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection, params=bind_params)


def prepare_weather_frame(raw: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {"date", "date_release", "year_market", "param", "geo", "value"}
    missing = expected_columns.difference(raw.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Weather data is missing required columns: {missing_text}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["date_release"] = pd.to_datetime(frame["date_release"]).dt.normalize()
    frame["year_market"] = pd.to_numeric(frame["year_market"], errors="coerce")
    frame["geo"] = frame["geo"].astype(str).str.lower()
    frame["country_code"] = frame["geo"].str[:3]
    frame["param"] = frame["param"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "date_release", "param", "geo", "value"])

    frame = frame.sort_values(["date", "date_release", "param", "geo"])
    frame = frame.groupby(["date", "geo", "param"], as_index=False).tail(1)
    return frame.reset_index(drop=True)


def reduce_to_country_level(
    prepared: pd.DataFrame,
    crop: CropDefinition,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    country_lookup = crop.country_lookup
    records: list[dict[str, object]] = []

    filtered = prepared[prepared["country_code"].isin(crop.country_codes)].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "date_release",
                "year_market",
                "country_code",
                "country_label",
                "iso_alpha3",
                "param",
                "value",
                "period",
            ]
        )

    for (date_value, country_code, param), group in filtered.groupby(
        ["date", "country_code", "param"], sort=True
    ):
        exact_country = group[group["geo"] == country_code]
        selected = exact_country if not exact_country.empty else group
        country = country_lookup[country_code]

        year_market_values = selected["year_market"].dropna()
        year_market = int(year_market_values.iloc[-1]) if not year_market_values.empty else None

        records.append(
            {
                "date": date_value,
                "date_release": selected["date_release"].max(),
                "year_market": year_market,
                "country_code": country.code,
                "country_label": country.label,
                "iso_alpha3": country.iso_alpha3,
                "param": param,
                "value": selected["value"].mean(),
            }
        )

    country_daily = pd.DataFrame.from_records(records)
    country_daily["period"] = np.where(
        country_daily["date"] > current_date,
        "forecast",
        "actual",
    )
    return country_daily.sort_values(["date", "country_code", "param"]).reset_index(drop=True)


def generate_sample_weather(crop: CropDefinition) -> pd.DataFrame:
    as_of = pd.Timestamp(
        os.getenv(
            "WEATHER_SAMPLE_AS_OF",
            pd.Timestamp.today().normalize().date().isoformat(),
        )
    )
    start_date = pd.Timestamp("2005-01-01")
    forecast_horizon = 15
    dates = pd.date_range(start_date, as_of + pd.Timedelta(days=forecast_horizon), freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    years_since_start = (dates.year - start_date.year).to_numpy()
    rng = np.random.default_rng(14)

    records: list[dict[str, object]] = []
    for country in crop.countries:
        phase_shift = 18 if country.code == "mys" else 0
        seasonal_cycle = 0.55 * np.sin((2 * np.pi * (day_of_year + phase_shift)) / 365.25)
        sub_seasonal_cycle = 0.25 * np.cos((4 * np.pi * (day_of_year + phase_shift)) / 365.25)
        climate_trend = years_since_start * 0.018
        recent_heat_pulse = np.where(
            (dates >= as_of - pd.Timedelta(days=10)) & (dates <= as_of + pd.Timedelta(days=5)),
            0.85 if country.code == "idn" else 0.45,
            0.0,
        )
        forecast_cooling = np.where(
            dates > as_of,
            np.linspace(0.2, -0.25, len(dates)),
            0.0,
        )

        mean_noise = rng.normal(loc=0.0, scale=0.28, size=len(dates))
        mean_temp = (
            (27.15 if country.code == "idn" else 26.65)
            + seasonal_cycle
            + sub_seasonal_cycle
            + climate_trend
            + recent_heat_pulse
            + forecast_cooling
            + mean_noise
        )
        max_temp = mean_temp + 4.2 + rng.normal(loc=0.0, scale=0.24, size=len(dates))
        min_temp = mean_temp - 4.0 + rng.normal(loc=0.0, scale=0.20, size=len(dates))

        series_by_param = {
            "palmoil-t2m_mean-degree_c": mean_temp,
            "palmoil-t2m_max-degree_c": max_temp,
            "palmoil-t2m_min-degree_c": min_temp,
        }

        for param, values in series_by_param.items():
            for date_value, value in zip(dates, values, strict=True):
                if date_value <= as_of:
                    records.append(
                        {
                            "date": date_value,
                            "date_release": date_value,
                            "year_market": date_value.year,
                            "param": param,
                            "geo": country.code,
                            "value": float(value),
                        }
                    )
                else:
                    records.append(
                        {
                            "date": date_value,
                            "date_release": as_of - pd.Timedelta(days=1),
                            "year_market": date_value.year,
                            "param": param,
                            "geo": country.code,
                            "value": float(value - 0.12),
                        }
                    )
                    records.append(
                        {
                            "date": date_value,
                            "date_release": as_of,
                            "year_market": date_value.year,
                            "param": param,
                            "geo": country.code,
                            "value": float(value),
                        }
                    )

    return pd.DataFrame.from_records(records)
