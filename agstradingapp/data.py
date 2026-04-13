from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CropDefinition, PALM_OIL
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
    if csv_path is None:
        raise missing_feed_error(crop)
    file_signature = csv_file_signature(csv_path)
    return _load_dataset_cached(crop, str(csv_path), file_signature)


@lru_cache(maxsize=8)
def _load_dataset_cached(
    crop: CropDefinition,
    csv_path_text: str,
    file_signature: tuple[int, int],
) -> CropDataset:
    del file_signature

    csv_path = Path(csv_path_text)
    raw = load_csv_weather(csv_path)
    source_mode = "csv"
    status_message = f"Loaded CSV feed from {describe_csv_path(csv_path)}."

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


def crop_csv_candidates(crop: CropDefinition) -> tuple[Path, ...]:
    override = os.getenv("AGSTRADINGAPP_DATA_FILE")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))

    candidates.append(data_dir() / crop.data_filename)
    candidates.append(repo_root() / crop.data_filename)

    cwd_candidate = Path.cwd() / crop.data_filename
    if cwd_candidate not in candidates:
        candidates.append(cwd_candidate)

    return tuple(candidates)


def crop_csv_path(crop: CropDefinition) -> Path | None:
    for candidate in crop_csv_candidates(crop):
        if candidate.exists():
            return candidate.resolve()
    return None


def csv_file_signature(csv_path: Path | None) -> tuple[int, int] | tuple[()]:
    if csv_path is None:
        return ()
    stat = csv_path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def describe_csv_path(csv_path: Path) -> str:
    try:
        return str(csv_path.relative_to(repo_root()))
    except ValueError:
        return str(csv_path)


def missing_feed_error(crop: CropDefinition) -> FileNotFoundError:
    candidate_lines = "\n".join(f"- {path}" for path in crop_csv_candidates(crop))
    return FileNotFoundError(
        "Palm oil CSV feed not found. Expected one of:\n"
        f"{candidate_lines}"
    )


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
