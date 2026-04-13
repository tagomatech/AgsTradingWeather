from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CountryDefinition, CropDefinition, PALM_OIL
from .domain import build_param_dictionary


@dataclass(frozen=True)
class CropDataset:
    crop: CropDefinition
    raw: pd.DataFrame
    geo_daily: pd.DataFrame
    country_daily: pd.DataFrame
    param_dictionary: pd.DataFrame
    current_date: pd.Timestamp
    source_mode: str
    status_message: str


def load_dataset(crop: CropDefinition = PALM_OIL) -> CropDataset:
    csv_path = crop_csv_path(crop)
    if csv_path.exists():
        raw = load_csv_weather(csv_path)
        source_mode = "csv"
        status_message = f"Loaded local CSV feed from data/{csv_path.name}."
    else:
        raw = generate_sample_weather(crop)
        source_mode = "sample"
        status_message = (
            f"Using deterministic sample data. Drop {crop.data_filename} into data/ to use "
            "the local feed."
        )

    prepared = prepare_weather_frame(raw)
    current_date = prepared["date_release"].max().normalize()
    geo_daily = enrich_geo_daily(prepared, crop=crop, current_date=current_date)
    country_daily = reduce_to_country_level(geo_daily, crop=crop, current_date=current_date)
    param_dictionary = build_param_dictionary(geo_daily["param"].unique())

    return CropDataset(
        crop=crop,
        raw=prepared,
        geo_daily=geo_daily,
        country_daily=country_daily,
        param_dictionary=param_dictionary,
        current_date=current_date,
        source_mode=source_mode,
        status_message=status_message,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    path = repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def crop_csv_path(crop: CropDefinition) -> Path:
    override = os.getenv("AGSTRADINGAPP_DATA_FILE")
    return Path(override) if override else data_dir() / crop.data_filename


def load_csv_weather(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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


def empty_geo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "date_release",
            "year_market",
            "country_code",
            "country_label",
            "iso_alpha3",
            "geo",
            "geo_label",
            "geo_level",
            "geo_weight",
            "param",
            "value",
            "period",
        ]
    )


def enrich_geo_daily(
    prepared: pd.DataFrame,
    crop: CropDefinition,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    available_geos = set(crop.all_geo_codes)
    country_lookup = crop.country_lookup
    region_lookup = crop.region_lookup
    region_country_lookup = crop.region_country_lookup

    filtered = prepared[prepared["geo"].isin(available_geos)].copy()
    if filtered.empty:
        return empty_geo_frame()

    geo_level: list[str] = []
    geo_label: list[str] = []
    geo_weight: list[float] = []
    country_label: list[str] = []
    iso_alpha3: list[str] = []

    for geo in filtered["geo"]:
        if geo in country_lookup:
            country = country_lookup[geo]
            geo_level.append("country")
            geo_label.append(country.label)
            geo_weight.append(100.0)
        else:
            region = region_lookup[geo]
            country = region_country_lookup[geo]
            geo_level.append("region")
            geo_label.append(region.label)
            geo_weight.append(region.weight)

        country_label.append(country.label)
        iso_alpha3.append(country.iso_alpha3)

    filtered["country_label"] = country_label
    filtered["iso_alpha3"] = iso_alpha3
    filtered["geo_level"] = geo_level
    filtered["geo_label"] = geo_label
    filtered["geo_weight"] = geo_weight
    filtered["period"] = np.where(filtered["date"] > current_date, "forecast", "actual")

    return filtered[
        [
            "date",
            "date_release",
            "year_market",
            "country_code",
            "country_label",
            "iso_alpha3",
            "geo",
            "geo_label",
            "geo_level",
            "geo_weight",
            "param",
            "value",
            "period",
        ]
    ].sort_values(["date", "geo", "param"]).reset_index(drop=True)


def reduce_to_country_level(
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    if geo_daily.empty:
        return empty_geo_frame()

    records: list[dict[str, object]] = []
    for (date_value, country_code, param), group in geo_daily.groupby(
        ["date", "country_code", "param"],
        sort=True,
    ):
        exact = group[group["geo"] == country_code]
        country = crop.country_lookup[country_code]

        if not exact.empty:
            selected = exact.iloc[-1]
            records.append(
                {
                    "date": selected["date"],
                    "date_release": selected["date_release"],
                    "year_market": selected["year_market"],
                    "country_code": country.code,
                    "country_label": country.label,
                    "iso_alpha3": country.iso_alpha3,
                    "geo": country.code,
                    "geo_label": country.label,
                    "geo_level": "country",
                    "geo_weight": 100.0,
                    "param": param,
                    "value": float(selected["value"]),
                    "period": "forecast" if selected["date"] > current_date else "actual",
                }
            )
            continue

        regional = group[group["geo_level"] == "region"].copy()
        if regional.empty:
            continue

        weights = regional["geo_weight"].astype(float)
        weighted_value = (
            float(np.average(regional["value"], weights=weights))
            if float(weights.sum()) > 0
            else float(regional["value"].mean())
        )
        records.append(
            {
                "date": date_value,
                "date_release": regional["date_release"].max(),
                "year_market": regional["year_market"].dropna().iloc[-1]
                if not regional["year_market"].dropna().empty
                else None,
                "country_code": country.code,
                "country_label": country.label,
                "iso_alpha3": country.iso_alpha3,
                "geo": country.code,
                "geo_label": country.label,
                "geo_level": "country",
                "geo_weight": 100.0,
                "param": param,
                "value": weighted_value,
                "period": "forecast" if date_value > current_date else "actual",
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["date", "geo", "param"]
    ).reset_index(drop=True)


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

    country_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    records: list[dict[str, object]] = []

    for country_index, country in enumerate(crop.countries):
        region_arrays: dict[str, dict[str, np.ndarray]] = {}
        for region_index, region in enumerate(country.regions):
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

            regional_wave = 0.18 * np.sin(
                (2 * np.pi * (day_of_year + (region_index + 1) * 11 + country_index * 7)) / 365.25
            )
            regional_offset = ((region_index % 5) - 2) * 0.12
            deterministic_noise = 0.08 * np.cos(
                (2 * np.pi * (day_of_year + (region_index + 3) * 17)) / 31.0
            )

            mean_temp = (
                (27.15 if country.code == "idn" else 26.65)
                + seasonal_cycle
                + sub_seasonal_cycle
                + climate_trend
                + recent_heat_pulse
                + forecast_cooling
                + regional_wave
                + regional_offset
                + deterministic_noise
            )
            max_temp = mean_temp + 4.2 + 0.18 * np.sin((2 * np.pi * day_of_year) / 29.0)
            min_temp = mean_temp - 4.0 + 0.16 * np.cos((2 * np.pi * day_of_year) / 23.0)

            region_arrays[region.geo] = {
                "palmoil-t2m_mean-degree_c": mean_temp,
                "palmoil-t2m_max-degree_c": max_temp,
                "palmoil-t2m_min-degree_c": min_temp,
            }

            for param, values in region_arrays[region.geo].items():
                for date_value, value in zip(dates, values, strict=True):
                    if date_value <= as_of:
                        records.append(
                            {
                                "date": date_value,
                                "date_release": date_value,
                                "year_market": date_value.year,
                                "param": param,
                                "geo": region.geo,
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
                                "geo": region.geo,
                                "value": float(value - 0.12),
                            }
                        )
                        records.append(
                            {
                                "date": date_value,
                                "date_release": as_of,
                                "year_market": date_value.year,
                                "param": param,
                                "geo": region.geo,
                                "value": float(value),
                            }
                        )

        weights = np.array([region.weight for region in country.regions], dtype=float)
        for param in crop.params:
            stacked = np.vstack([region_arrays[region.geo][param] for region in country.regions])
            country_arrays[(country.code, param)] = {
                "current": np.average(stacked, axis=0, weights=weights),
                "previous": np.average(stacked - 0.12, axis=0, weights=weights),
            }

    for country in crop.countries:
        for param in crop.params:
            payload = country_arrays[(country.code, param)]
            for idx, date_value in enumerate(dates):
                if date_value <= as_of:
                    records.append(
                        {
                            "date": date_value,
                            "date_release": date_value,
                            "year_market": date_value.year,
                            "param": param,
                            "geo": country.code,
                            "value": float(payload["current"][idx]),
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
                            "value": float(payload["previous"][idx]),
                        }
                    )
                    records.append(
                        {
                            "date": date_value,
                            "date_release": as_of,
                            "year_market": date_value.year,
                            "param": param,
                            "geo": country.code,
                            "value": float(payload["current"][idx]),
                        }
                    )

    return pd.DataFrame.from_records(records)
