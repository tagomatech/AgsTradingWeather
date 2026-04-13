from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CropDefinition
from .domain import STATISTIC_ORDER, parse_param_label


def is_country_scope(scope: str, crop: CropDefinition) -> bool:
    return scope in crop.country_codes


def is_region_scope(scope: str, crop: CropDefinition) -> bool:
    return scope in crop.region_lookup


def filter_snapshot(snapshot: pd.DataFrame, scope: str, crop: CropDefinition) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    if scope == "all":
        return snapshot[snapshot["geo_level"] == "country"].copy()
    if is_country_scope(scope, crop):
        return snapshot[snapshot["country_code"] == scope].copy()
    return snapshot[snapshot["geo"] == scope].copy()


def build_snapshot(geo_daily: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    actual = geo_daily[geo_daily["date"] <= current_date].copy()
    if actual.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for (geo, param), group in actual.groupby(["geo", "param"], sort=True):
        group = group.sort_values("date")
        latest_date = group["date"].max()
        current_window = group[group["date"].between(latest_date - pd.Timedelta(days=6), latest_date)]
        history = group[group["date"].dt.year < latest_date.year].copy()

        descriptor = parse_param_label(param)
        month_day_keys = current_window["date"].dt.strftime("%m-%d")
        reference_distribution = (
            history.assign(month_day=history["date"].dt.strftime("%m-%d"))
            .loc[lambda frame: frame["month_day"].isin(month_day_keys)]
            .groupby(history["date"].dt.year)["value"]
            .mean()
        )

        latest_value = float(group.loc[group["date"] == latest_date, "value"].mean())
        trailing_mean = float(current_window["value"].mean())
        climatology_mean = (
            float(reference_distribution.mean()) if not reference_distribution.empty else math.nan
        )
        climatology_std = (
            float(reference_distribution.std(ddof=0))
            if len(reference_distribution) > 1
            else math.nan
        )
        climatology_std = max(climatology_std, 0.18) if not math.isnan(climatology_std) else 0.18
        anomaly = trailing_mean - climatology_mean if not math.isnan(climatology_mean) else math.nan
        zscore = anomaly / climatology_std if not math.isnan(anomaly) else math.nan
        percentile = (
            float((reference_distribution < trailing_mean).mean() * 100)
            if not reference_distribution.empty
            else math.nan
        )

        records.append(
            {
                "geo": geo,
                "geo_label": group["geo_label"].iloc[0],
                "geo_level": group["geo_level"].iloc[0],
                "geo_weight": float(group["geo_weight"].iloc[0]),
                "country_code": group["country_code"].iloc[0],
                "country_label": group["country_label"].iloc[0],
                "iso_alpha3": group["iso_alpha3"].iloc[0],
                "param": param,
                "metric_label": descriptor.short_label,
                "latest_value": latest_value,
                "trailing_7d_mean": trailing_mean,
                "climatology_mean": climatology_mean,
                "anomaly": anomaly,
                "zscore": zscore,
                "abs_zscore": abs(zscore) if not math.isnan(zscore) else math.nan,
                "percentile": percentile,
                "issue_flag": classify_issue(zscore),
                "unit_label": descriptor.unit_label,
                "stat_order": STATISTIC_ORDER.get(descriptor.statistic_code, 99),
                "level_order": 0 if group["geo_level"].iloc[0] == "country" else 1,
            }
        )

    snapshot = pd.DataFrame.from_records(records)
    if snapshot.empty:
        return snapshot

    return snapshot.sort_values(
        ["country_label", "level_order", "geo_weight", "stat_order", "metric_label"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)


def classify_issue(zscore: float) -> str:
    if math.isnan(zscore):
        return "No signal"
    if zscore >= 1.8:
        return "Heat stress"
    if zscore >= 1.0:
        return "Warm watch"
    if zscore <= -1.8:
        return "Cool stress"
    if zscore <= -1.0:
        return "Cool watch"
    return "Normal"


def build_kpi_summary(
    snapshot: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
) -> list[dict[str, str]]:
    scoped = filter_snapshot(snapshot, scope, crop)
    if scoped.empty:
        return []

    warm_count = int(scoped["zscore"].ge(1.0).sum())
    cool_count = int(scoped["zscore"].le(-1.0).sum())
    top_signal = scoped.sort_values("abs_zscore", ascending=False).iloc[0]

    if scope == "all":
        coverage = f"{len(crop.countries)} country cuts / {scoped['param'].nunique()} metrics"
        coverage_detail = "Core belt overview"
    elif is_country_scope(scope, crop):
        coverage = f"{scoped['geo'].nunique()} geos / {scoped['param'].nunique()} metrics"
        coverage_detail = crop.country_lookup[scope].label
    else:
        coverage = f"{scoped['geo'].nunique()} geo / {scoped['param'].nunique()} metrics"
        coverage_detail = top_signal["country_label"]

    top_signal_text = (
        f"{top_signal['geo_label']} {top_signal['metric_label']} "
        f"{top_signal['anomaly']:+.2f} {top_signal['unit_label']}"
    )

    return [
        {"label": "Coverage", "value": coverage, "detail": coverage_detail},
        {
            "label": "Current Cut",
            "value": current_date.strftime("%d %b %Y"),
            "detail": "latest release date",
        },
        {
            "label": "Warm / Cool Signals",
            "value": f"{warm_count} / {cool_count}",
            "detail": "z-score beyond +/-1.0",
        },
        {
            "label": "Largest Deviation",
            "value": top_signal["issue_flag"],
            "detail": top_signal_text,
        },
    ]


def aggregate_scope_series(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    if scope == "all":
        weights = crop.country_weights
        scoped = country_daily[country_daily["country_code"].isin(weights)].copy()
        if scoped.empty:
            return scoped, f"{crop.label} core belt"

        scoped["weight"] = scoped["country_code"].map(weights)
        records: list[dict[str, object]] = []
        for (date_value, param), group in scoped.groupby(["date", "param"], sort=True):
            available_weight = float(group["weight"].sum())
            weighted_value = (
                float(np.average(group["value"], weights=group["weight"]))
                if available_weight > 0
                else float(group["value"].mean())
            )
            records.append(
                {
                    "date": date_value,
                    "date_release": group["date_release"].max(),
                    "param": param,
                    "value": weighted_value,
                    "period": "forecast" if date_value > current_date else "actual",
                }
            )
        return pd.DataFrame.from_records(records), f"{crop.label} core belt"

    if is_country_scope(scope, crop):
        scoped = country_daily[country_daily["country_code"] == scope].copy()
        label = crop.country_lookup[scope].label if not scoped.empty else scope.upper()
        return scoped, label

    scoped = geo_daily[geo_daily["geo"] == scope].copy()
    if scoped.empty:
        return scoped, scope
    return scoped, scoped["geo_label"].iloc[0]


def build_recent_context(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
    lookback_days: int = 120,
    forward_days: int = 15,
) -> tuple[pd.DataFrame, str]:
    scoped, scope_label = aggregate_scope_series(
        country_daily=country_daily,
        geo_daily=geo_daily,
        crop=crop,
        scope=scope,
        current_date=current_date,
    )
    if scoped.empty:
        return scoped, scope_label

    window_start = current_date - pd.Timedelta(days=lookback_days)
    window_end = current_date + pd.Timedelta(days=forward_days)
    scoped = scoped.copy()
    scoped["month_day"] = scoped["date"].dt.strftime("%m-%d")

    reference = (
        scoped[scoped["date"].dt.year < current_date.year]
        .groupby(["param", "month_day"])["value"]
        .agg(
            clim_mean="mean",
            clim_q10=lambda series: series.quantile(0.10),
            clim_q90=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )

    recent = scoped[scoped["date"].between(window_start, window_end)].copy()
    recent = recent.merge(reference, on=["param", "month_day"], how="left")
    return recent.sort_values(["param", "date"]).reset_index(drop=True), scope_label


def build_monthly_issue_matrix(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
    months: int = 15,
) -> pd.DataFrame:
    scoped, scope_label = aggregate_scope_series(
        country_daily=country_daily,
        geo_daily=geo_daily,
        crop=crop,
        scope=scope,
        current_date=current_date,
    )
    actual = scoped[scoped["date"] <= current_date].copy()
    if actual.empty:
        return pd.DataFrame()

    month_starts = pd.date_range(
        end=current_date.normalize().replace(day=1),
        periods=months,
        freq="MS",
    )

    records: list[dict[str, object]] = []
    for _, group in actual.groupby(["param"], sort=True):
        descriptor = parse_param_label(group["param"].iloc[0])
        for month_start in month_starts:
            month_end = month_start + pd.offsets.MonthEnd(0)
            current_month = group[group["date"].between(month_start, month_end)]
            historical_months = (
                group[
                    (group["date"] < month_start)
                    & (group["date"].dt.month == month_start.month)
                ]
                .groupby(group["date"].dt.year)["value"]
                .mean()
            )

            current_value = float(current_month["value"].mean()) if not current_month.empty else math.nan
            climatology_mean = (
                float(historical_months.mean()) if not historical_months.empty else math.nan
            )
            climatology_std = (
                float(historical_months.std(ddof=0))
                if len(historical_months) > 1
                else math.nan
            )
            climatology_std = max(climatology_std, 0.18) if not math.isnan(climatology_std) else 0.18
            anomaly = current_value - climatology_mean if not math.isnan(current_value) else math.nan
            zscore = anomaly / climatology_std if not math.isnan(anomaly) else math.nan

            records.append(
                {
                    "scope_label": scope_label,
                    "row_label": descriptor.short_label,
                    "unit_label": descriptor.unit_label,
                    "month_label": month_start.strftime("%b\n%Y"),
                    "month_start": month_start,
                    "zscore": zscore,
                    "anomaly": anomaly,
                    "display_value": f"{anomaly:+.2f}" if not math.isnan(anomaly) else "",
                }
            )

    matrix = pd.DataFrame.from_records(records)
    if matrix.empty:
        return matrix
    return matrix.sort_values(["row_label", "month_start"]).reset_index(drop=True)
