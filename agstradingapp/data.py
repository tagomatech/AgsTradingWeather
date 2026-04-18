from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import build_snapshot
from .config import CropDefinition, PALM_OIL
from .domain import build_param_dictionary

CACHE_VERSION = "v3"


@dataclass(frozen=True)
class CropDataset:
    crop: CropDefinition
    raw: pd.DataFrame
    geo_daily: pd.DataFrame
    country_daily: pd.DataFrame
    core_belt_daily: pd.DataFrame
    param_dictionary: pd.DataFrame
    snapshot: pd.DataFrame
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
    csv_path = Path(csv_path_text)
    source_mode = "csv"
    status_message = f"Loaded CSV feed from {describe_csv_path(csv_path)}."
    cached_payload = read_prepared_cache(crop, file_signature)

    if cached_payload is None:
        raw = load_csv_weather(csv_path)
        prepared = prepare_weather_frame(raw)
        current_date = prepared["date_release"].max().normalize()
        geo_daily = enrich_geo_daily(prepared, crop=crop, current_date=current_date)
        country_daily = reduce_to_country_level(geo_daily, crop=crop, current_date=current_date)
        core_belt_daily = build_core_belt_daily(country_daily, crop=crop, current_date=current_date)
        param_dictionary = build_param_dictionary(geo_daily["param"].unique())
        snapshot = build_snapshot(geo_daily, current_date)
        cached_payload = {
            "raw": prepared,
            "geo_daily": geo_daily,
            "country_daily": country_daily,
            "core_belt_daily": core_belt_daily,
            "param_dictionary": param_dictionary,
            "snapshot": snapshot,
            "current_date": current_date,
        }
        write_prepared_cache(crop, file_signature, cached_payload)
    else:
        prepared = cached_payload["raw"]
        geo_daily = cached_payload["geo_daily"]
        country_daily = cached_payload["country_daily"]
        core_belt_daily = cached_payload["core_belt_daily"]
        param_dictionary = cached_payload["param_dictionary"]
        snapshot = cached_payload["snapshot"]
        current_date = cached_payload["current_date"]

    return CropDataset(
        crop=crop,
        raw=prepared,
        geo_daily=geo_daily,
        country_daily=country_daily,
        core_belt_daily=core_belt_daily,
        param_dictionary=param_dictionary,
        snapshot=snapshot,
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


def cache_dir() -> Path:
    path = data_dir() / ".cache"
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


def prepared_cache_path(crop: CropDefinition, file_signature: tuple[int, int]) -> Path:
    mtime_ns, size = file_signature
    filename = f"{crop.crop_id}_{CACHE_VERSION}_{mtime_ns}_{size}.pkl"
    return cache_dir() / filename


def read_prepared_cache(
    crop: CropDefinition,
    file_signature: tuple[int, int],
) -> dict[str, object] | None:
    if not file_signature:
        return None

    path = prepared_cache_path(crop, file_signature)
    if not path.exists():
        return None

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        path.unlink(missing_ok=True)
        return None

    if not isinstance(payload, dict):
        path.unlink(missing_ok=True)
        return None
    return payload


def prune_prepared_cache(crop: CropDefinition, keep_path: Path) -> None:
    pattern = f"{crop.crop_id}_{CACHE_VERSION}_*.pkl"
    for path in cache_dir().glob(pattern):
        if path != keep_path:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                continue


def write_prepared_cache(
    crop: CropDefinition,
    file_signature: tuple[int, int],
    payload: dict[str, object],
) -> None:
    if not file_signature:
        return

    path = prepared_cache_path(crop, file_signature)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    prune_prepared_cache(crop, path)


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
    return pd.read_csv(
        csv_path,
        usecols=["date", "date_release", "year_market", "param", "geo", "value"],
        parse_dates=["date", "date_release"],
        dtype={
            "year_market": "string",
            "param": "string",
            "geo": "string",
            "value": "float64",
        },
        low_memory=False,
    )


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
    frame = frame.drop_duplicates(subset=["date", "geo", "param"], keep="last")
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

    geo_label_lookup = {country.code: country.label for country in crop.countries}
    geo_label_lookup.update({region.geo: region.label for region in region_lookup.values()})

    geo_weight_lookup = {country.code: 100.0 for country in crop.countries}
    geo_weight_lookup.update({region.geo: region.weight for region in region_lookup.values()})

    filtered["country_label"] = filtered["country_code"].map(
        {code: country.label for code, country in country_lookup.items()}
    )
    filtered["iso_alpha3"] = filtered["country_code"].map(
        {code: country.iso_alpha3 for code, country in country_lookup.items()}
    )
    filtered["geo_level"] = np.where(filtered["geo"].isin(country_lookup), "country", "region")
    filtered["geo_label"] = filtered["geo"].map(geo_label_lookup)
    filtered["geo_weight"] = filtered["geo"].map(geo_weight_lookup).astype(float)
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

    target_columns = list(empty_geo_frame().columns)

    exact = geo_daily[geo_daily["geo"] == geo_daily["country_code"]].copy()
    if not exact.empty:
        exact["geo"] = exact["country_code"]
        exact["geo_label"] = exact["country_label"]
        exact["geo_level"] = "country"
        exact["geo_weight"] = 100.0
        exact["period"] = np.where(exact["date"] > current_date, "forecast", "actual")

    regional = geo_daily[geo_daily["geo_level"] == "region"].copy()
    if regional.empty:
        return exact[target_columns].sort_values(["date", "geo", "param"]).reset_index(drop=True)

    regional["weighted_component"] = regional["value"] * regional["geo_weight"]
    aggregated = (
        regional.groupby(["date", "country_code", "param"], as_index=False)
        .agg(
            date_release=("date_release", "max"),
            year_market=("year_market", "max"),
            country_label=("country_label", "first"),
            iso_alpha3=("iso_alpha3", "first"),
            weighted_component=("weighted_component", "sum"),
            weight_total=("geo_weight", "sum"),
            mean_value=("value", "mean"),
        )
        .copy()
    )
    aggregated["value"] = np.where(
        aggregated["weight_total"] > 0,
        aggregated["weighted_component"] / aggregated["weight_total"],
        aggregated["mean_value"],
    )
    aggregated["geo"] = aggregated["country_code"]
    aggregated["geo_label"] = aggregated["country_label"]
    aggregated["geo_level"] = "country"
    aggregated["geo_weight"] = 100.0
    aggregated["period"] = np.where(aggregated["date"] > current_date, "forecast", "actual")

    if not exact.empty:
        exact_keys = exact[["date", "country_code", "param"]].drop_duplicates()
        aggregated = aggregated.merge(
            exact_keys.assign(has_exact=True),
            on=["date", "country_code", "param"],
            how="left",
        )
        aggregated = aggregated[aggregated["has_exact"] != True].drop(columns=["has_exact"])

    aggregated = aggregated[target_columns]
    combined = pd.concat([exact[target_columns], aggregated], ignore_index=True)
    return combined.sort_values(["date", "geo", "param"]).reset_index(drop=True)


def build_core_belt_daily(
    country_daily: pd.DataFrame,
    crop: CropDefinition,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    if country_daily.empty:
        return pd.DataFrame(columns=["date", "date_release", "param", "value", "period"])

    weights = pd.Series(crop.country_weights, name="country_weight", dtype="float64")
    scoped = country_daily[country_daily["country_code"].isin(weights.index)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["date", "date_release", "param", "value", "period"])

    scoped["country_weight"] = scoped["country_code"].map(weights)
    scoped["weighted_component"] = scoped["value"] * scoped["country_weight"]
    aggregated = (
        scoped.groupby(["date", "param"], as_index=False)
        .agg(
            date_release=("date_release", "max"),
            weighted_component=("weighted_component", "sum"),
            weight_total=("country_weight", "sum"),
            mean_value=("value", "mean"),
        )
        .copy()
    )
    aggregated["value"] = np.where(
        aggregated["weight_total"] > 0,
        aggregated["weighted_component"] / aggregated["weight_total"],
        aggregated["mean_value"],
    )
    aggregated["period"] = np.where(aggregated["date"] > current_date, "forecast", "actual")
    return aggregated[["date", "date_release", "param", "value", "period"]].sort_values(
        ["date", "param"]
    ).reset_index(drop=True)
