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


def describe_scope(scope: str, crop: CropDefinition) -> str:
    if scope == "all":
        return f"{crop.label} countries"
    if is_country_scope(scope, crop):
        return crop.country_lookup[scope].label
    region = crop.region_lookup.get(scope)
    if region is None:
        return scope
    country = crop.region_country_lookup[scope]
    return f"{country.label} / {region.label}"


def describe_comparison_mode(comparison_mode: str, window_days: int) -> str:
    if comparison_mode == "previous_window":
        return f"previous {window_days} days"
    if comparison_mode == "seasonal_normal":
        return "historical normal"
    return "same dates last year"


def resolve_precipitation_param(crop: CropDefinition) -> str | None:
    for param in crop.params:
        if parse_param_label(param).signal_family == "precipitation":
            return param
    return None


def aggregate_window_value(values: pd.Series, statistic_code: str) -> float:
    if values.empty:
        return math.nan
    if statistic_code == "sum":
        return float(values.sum())
    return float(values.mean())


def rolling_reference_distribution(
    values: pd.Series,
    window_days: int,
    statistic_code: str,
) -> pd.Series:
    ordered = values.sort_index()
    if statistic_code == "sum":
        return ordered.rolling(window_days).sum().dropna()
    return ordered.rolling(window_days).mean().dropna()


def build_snapshot(
    geo_daily: pd.DataFrame,
    current_date: pd.Timestamp,
    window_days: int = 7,
    comparison_mode: str = "same_period_last_year",
) -> pd.DataFrame:
    actual = geo_daily[geo_daily["date"] <= current_date].copy()
    if actual.empty:
        return pd.DataFrame()

    window_days = max(int(window_days), 1)
    records: list[dict[str, object]] = []
    for (geo, param), group in actual.groupby(["geo", "param"], sort=True):
        group = group.sort_values("date")
        latest_date = group["date"].max()
        window_start = latest_date - pd.Timedelta(days=window_days - 1)
        current_window = group[
            group["date"].between(window_start, latest_date)
        ]
        history = group[group["date"].dt.year < latest_date.year].copy()

        descriptor = parse_param_label(param)
        month_day_keys = current_window["date"].dt.strftime("%m-%d")
        seasonal_distribution = (
            history.assign(
                month_day=history["date"].dt.strftime("%m-%d"),
                year=history["date"].dt.year,
            )
            .loc[lambda frame: frame["month_day"].isin(month_day_keys)]
            .groupby("year")["value"]
            .agg("sum" if descriptor.statistic_code == "sum" else "mean")
        )

        reference_distribution = seasonal_distribution
        reference_label = describe_comparison_mode(comparison_mode, window_days)
        if comparison_mode == "previous_window":
            previous_end = window_start - pd.Timedelta(days=1)
            previous_start = previous_end - pd.Timedelta(days=window_days - 1)
            reference_window = group[group["date"].between(previous_start, previous_end)]
            reference_mean = (
                aggregate_window_value(reference_window["value"], descriptor.statistic_code)
                if len(reference_window) >= window_days
                else math.nan
            )
            reference_distribution = rolling_reference_distribution(
                group[group["date"] < window_start].sort_values("date")["value"],
                window_days,
                descriptor.statistic_code,
            )
        elif comparison_mode == "seasonal_normal":
            reference_mean = (
                float(seasonal_distribution.mean())
                if not seasonal_distribution.empty
                else math.nan
            )
        else:
            last_year_start = window_start - pd.DateOffset(years=1)
            last_year_end = latest_date - pd.DateOffset(years=1)
            reference_window = group[group["date"].between(last_year_start, last_year_end)]
            reference_mean = (
                aggregate_window_value(reference_window["value"], descriptor.statistic_code)
                if len(reference_window) >= window_days
                else math.nan
            )

        latest_value = float(group.loc[group["date"] == latest_date, "value"].mean())
        trailing_mean = aggregate_window_value(current_window["value"], descriptor.statistic_code)
        reference_std = (
            float(reference_distribution.std(ddof=0))
            if len(reference_distribution) > 1
            else math.nan
        )
        reference_std = max(reference_std, 0.18) if not math.isnan(reference_std) else 0.18
        anomaly = trailing_mean - reference_mean if not math.isnan(reference_mean) else math.nan
        zscore = anomaly / reference_std if not math.isnan(anomaly) else math.nan
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
                "window_mean": trailing_mean,
                "window_days": window_days,
                "reference_mean": reference_mean,
                "reference_label": reference_label,
                "anomaly": anomaly,
                "zscore": zscore,
                "abs_zscore": abs(zscore) if not math.isnan(zscore) else math.nan,
                "percentile": percentile,
                "history_years": int(len(reference_distribution)),
                "issue_flag": classify_issue(zscore, descriptor.signal_family),
                "unit_label": descriptor.unit_label,
                "comparison_mode": comparison_mode,
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


def classify_issue(zscore: float, signal_family: str = "other") -> str:
    if math.isnan(zscore):
        return "No signal"

    if signal_family == "precipitation":
        if zscore >= 1.8:
            return "Wet stress"
        if zscore >= 1.0:
            return "Wet watch"
        if zscore <= -1.8:
            return "Dry stress"
        if zscore <= -1.0:
            return "Dry watch"
        return "Normal"

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
    focus_param: str | None = None,
    window_days: int = 7,
    comparison_mode: str = "same_period_last_year",
) -> list[dict[str, str]]:
    scoped = filter_snapshot(snapshot, scope, crop)
    if focus_param:
        scoped = scoped[scoped["param"] == focus_param].copy()
    if scoped.empty:
        return []

    top_signal = scoped.sort_values("abs_zscore", ascending=False).iloc[0]
    metric_label = top_signal["metric_label"]
    scope_label = describe_scope(scope, crop)
    signal_family = parse_param_label(top_signal["param"]).signal_family

    positive_count = int(scoped["zscore"].ge(1.0).sum())
    negative_count = int(scoped["zscore"].le(-1.0).sum())
    balance_label = "Wet / Dry" if signal_family == "precipitation" else "Warm / Cool"

    top_signal_text = f"{top_signal['geo_label']} {top_signal['anomaly']:+.2f} {top_signal['unit_label']}"

    return [
        {"label": "Geo Focus", "value": scope_label, "detail": f"{scoped['geo'].nunique()} geo cuts"},
        {"label": "Metric", "value": metric_label, "detail": "selected weather lens"},
        {"label": "Signal Window", "value": f"{window_days} days", "detail": "used for anomaly and z-score"},
        {"label": "Compare To", "value": describe_comparison_mode(comparison_mode, window_days), "detail": "reference basis"},
        {
            "label": "Current Cut",
            "value": current_date.strftime("%d %b %Y"),
            "detail": "latest release date",
        },
        {
            "label": balance_label,
            "value": f"{positive_count} / {negative_count}",
            "detail": "z-score beyond +/-1.0",
        },
        {"label": "Top Signal", "value": top_signal["issue_flag"], "detail": top_signal_text},
    ]


def aggregate_scope_series(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
    core_belt_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    if scope == "all":
        scoped = core_belt_daily.copy() if core_belt_daily is not None else pd.DataFrame()
        return scoped, f"{crop.label} countries"

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
    core_belt_daily: pd.DataFrame | None = None,
    focus_param: str | None = None,
    lookback_days: int = 120,
    forward_days: int = 15,
    window_days: int = 7,
    comparison_mode: str = "same_period_last_year",
) -> tuple[pd.DataFrame, str]:
    scoped, scope_label = aggregate_scope_series(
        country_daily=country_daily,
        geo_daily=geo_daily,
        crop=crop,
        scope=scope,
        current_date=current_date,
        core_belt_daily=core_belt_daily,
    )
    if scoped.empty:
        return scoped, scope_label

    window_start = current_date - pd.Timedelta(days=lookback_days)
    window_end = current_date + pd.Timedelta(days=forward_days)
    scoped = scoped.copy().sort_values(["param", "date"])
    scoped["month_day"] = scoped["date"].dt.strftime("%m-%d")
    history = scoped[scoped["date"].dt.year < current_date.year].copy()
    climatology = (
        history.groupby(["param", "month_day"])["value"]
        .agg(
            climatology_mean="mean",
            reference_low=lambda series: series.quantile(0.10),
            reference_high=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )

    if comparison_mode == "previous_window":
        scoped["reference_mean"] = (
            scoped.groupby("param")["value"]
            .transform(lambda series: series.shift(1).rolling(window_days, min_periods=window_days).mean())
        )
        scoped["reference_low"] = np.nan
        scoped["reference_high"] = np.nan
    elif comparison_mode == "same_period_last_year":
        prior_year = history[["param", "date", "value"]].copy()
        prior_year["reference_date"] = prior_year["date"] + pd.DateOffset(years=1)
        prior_year = prior_year.rename(columns={"value": "reference_mean"})

        scoped = scoped.merge(
            prior_year[["param", "reference_date", "reference_mean"]],
            left_on=["param", "date"],
            right_on=["param", "reference_date"],
            how="left",
        ).drop(columns=["reference_date"])
        scoped = scoped.merge(climatology, on=["param", "month_day"], how="left")
    else:
        reference = climatology.rename(columns={"climatology_mean": "reference_mean"})
        scoped = scoped.merge(reference, on=["param", "month_day"], how="left")

    if "reference_mean" not in scoped.columns and "climatology_mean" in scoped.columns:
        scoped["reference_mean"] = scoped["climatology_mean"]
    elif "climatology_mean" in scoped.columns:
        scoped = scoped.drop(columns=["climatology_mean"])

    recent = scoped[scoped["date"].between(window_start, window_end)].copy()
    if focus_param:
        recent = recent[recent["param"] == focus_param].copy()
    return recent.sort_values(["param", "date"]).reset_index(drop=True), scope_label


def build_monthly_issue_matrix(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    current_date: pd.Timestamp,
    core_belt_daily: pd.DataFrame | None = None,
    focus_param: str | None = None,
    months: int = 15,
) -> pd.DataFrame:
    scoped, scope_label = aggregate_scope_series(
        country_daily=country_daily,
        geo_daily=geo_daily,
        crop=crop,
        scope=scope,
        current_date=current_date,
        core_belt_daily=core_belt_daily,
    )
    actual = scoped[scoped["date"] <= current_date].copy()
    if focus_param:
        actual = actual[actual["param"] == focus_param].copy()
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


def build_rainfall_threshold_matrix(
    country_daily: pd.DataFrame,
    geo_daily: pd.DataFrame,
    crop: CropDefinition,
    scope: str,
    months: int = 18,
) -> tuple[pd.DataFrame, str]:
    precip_param = resolve_precipitation_param(crop)
    if precip_param is None:
        return pd.DataFrame(), describe_scope(scope, crop)

    if scope == "all":
        scoped = country_daily[
            (country_daily["param"] == precip_param) & (country_daily["geo_level"] == "country")
        ].copy()
        row_order = list(crop.country_codes)
        row_labels = {country.code: country.label for country in crop.countries}
        scope_label = "Whole countries"
    elif is_country_scope(scope, crop):
        scoped = geo_daily[
            (geo_daily["param"] == precip_param)
            & (geo_daily["country_code"] == scope)
            & (geo_daily["geo_level"] == "region")
        ].copy()
        row_order = [region.geo for region in crop.country_lookup[scope].regions]
        row_labels = {region.geo: region.label for region in crop.country_lookup[scope].regions}
        scope_label = crop.country_lookup[scope].label
    else:
        country = crop.region_country_lookup[scope]
        scoped = geo_daily[
            (geo_daily["param"] == precip_param)
            & (geo_daily["country_code"] == country.code)
            & (geo_daily["geo_level"] == "region")
        ].copy()
        row_order = [region.geo for region in country.regions]
        row_labels = {
            region.geo: f"{region.label} *" if region.geo == scope else region.label
            for region in country.regions
        }
        scope_label = f"{country.label} regions"

    if scoped.empty:
        return pd.DataFrame(), scope_label

    scoped["month_start"] = scoped["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        scoped.groupby(["geo", "month_start"], as_index=False)
        .agg(
            total_mm=("value", "sum"),
            obs_days=("date", "nunique"),
        )
        .copy()
    )
    monthly["days_in_month"] = monthly["month_start"].dt.days_in_month
    monthly = monthly[monthly["obs_days"] >= monthly["days_in_month"]].copy()
    if monthly.empty:
        return pd.DataFrame(), scope_label

    month_order = sorted(monthly["month_start"].unique())[-months:]
    month_frame = pd.DataFrame({"month_start": month_order})
    row_frame = pd.DataFrame({"geo": row_order})
    base = row_frame.merge(month_frame, how="cross")

    matrix = base.merge(monthly, on=["geo", "month_start"], how="left")
    matrix["row_label"] = matrix["geo"].map(row_labels)
    matrix["bucket"] = np.select(
        [
            matrix["total_mm"] < 100,
            matrix["total_mm"] > 400,
        ],
        [-1, 1],
        default=0,
    )
    matrix.loc[matrix["total_mm"].isna(), "bucket"] = np.nan
    matrix["month_label"] = matrix["month_start"].dt.strftime("%b\n%Y")
    matrix["hover_value"] = matrix["total_mm"].map(
        lambda value: f"{value:.0f} mm" if pd.notna(value) else "No complete month"
    )

    return matrix.sort_values(
        ["geo", "month_start"],
        key=lambda series: series.map({geo: idx for idx, geo in enumerate(row_order)}) if series.name == "geo" else series,
    ).reset_index(drop=True), scope_label
