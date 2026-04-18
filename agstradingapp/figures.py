from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CropDefinition
from .domain import STATISTIC_ORDER, describe_window_measure, parse_param_label


PAPER_COLOR = "rgba(244, 240, 232, 0)"
PLOT_COLOR = "rgba(255, 255, 255, 0.94)"
GRID_COLOR = "rgba(74, 102, 112, 0.16)"
TEXT_COLOR = "#243740"
ACCENT_COLOR = "#1b6b7a"
FORECAST_COLOR = "#c67a2d"
BAND_COLOR = "rgba(84, 109, 120, 0.12)"
DIVERGING_SCALE = ["#4d8ab0", "#f5efe5", "#c56e4d"]
SEQUENTIAL_SCALE = ["#e2eef1", "#85a9b4", "#244c58"]
REGION_BOUNDARY_NAME = {
    "idn-riau": "Riau",
    "idn-kalimantan_tengah": "Central Kalimantan",
    "idn-kalimantan_barat": "West Kalimantan",
    "idn-kalimantan_timur": "East Kalimantan",
    "idn-sumatera_selatan": "South Sumatra",
    "idn-sumatera_utara": "North Sumatra",
    "idn-jambi": "Jambi",
    "idn-kalimantan_selatan": "South Kalimantan",
    "idn-aceh": "Aceh",
    "idn-bangka_belitung": "Bangka-Belitung Islands",
    "idn-bengkulu": "Bengkulu",
    "idn-sumatera_barat": "West Sumatra",
    "idn-lampung": "Lampung",
    "idn-kalimantan_utara": "North Kalimantan",
    "idn-papua": "Papua",
    "mys-sabah": "Sabah",
    "mys-sarawak": "Sarawak",
    "mys-johor": "Johor",
    "mys-pahang": "Pahang",
    "mys-perak": "Perak",
    "mys-negeri_sembilan": "Negeri Sembilan",
    "mys-selangor": "Selangor",
    "mys-trengganu": "Terengganu",
    "mys-kelantan": "Kelantan",
    "mys-kedah": "Kedah",
    "mys-melaka": "Malacca",
}
REGIONAL_GEOJSON_FILES = {
    "IDN": Path(__file__).resolve().parent.parent / "data" / "geo" / "idn_adm1_simplified.geojson",
    "MYS": Path(__file__).resolve().parent.parent / "data" / "geo" / "mys_adm1_simplified.geojson",
}


def safe_percent_of_normal(values: pd.Series, references: pd.Series) -> pd.Series:
    denominator = references.astype(float)
    numerator = values.astype(float)
    return pd.Series(
        np.where(denominator.abs() > 1e-9, (numerator / denominator) * 100.0, np.nan),
        index=values.index,
        dtype="float64",
    )


def geo_display_spec(frame: pd.DataFrame, descriptor, display_mode: str) -> dict[str, object]:
    series = frame["zscore"]
    title = "Signal z-score"
    axis_title = "z-score"
    color_title = "z-score"
    reference_line = 0.0
    colorscale = DIVERGING_SCALE
    color_series = frame["zscore"]
    range_color = (-2.5, 2.5)

    if display_mode == "anomaly":
        series = frame["anomaly"]
        limit = max(0.5, float(series.abs().quantile(0.95))) if not series.dropna().empty else 1.0
        title = f"Anomaly ({descriptor.unit_label})"
        axis_title = f"Anomaly ({descriptor.unit_label})"
        color_title = descriptor.unit_label
        reference_line = 0.0
        colorscale = DIVERGING_SCALE
        color_series = series
        range_color = (-limit, limit)
    elif display_mode == "pct_normal":
        series = frame["pct_normal"]
        delta = frame["pct_normal"] - 100.0
        limit = max(10.0, float(delta.abs().quantile(0.95))) if not delta.dropna().empty else 25.0
        title = "% of normal"
        axis_title = "% of normal"
        color_title = "% of normal"
        reference_line = 100.0
        colorscale = DIVERGING_SCALE
        color_series = delta
        range_color = (-limit, limit)
    elif display_mode == "actual_level":
        series = frame["window_mean"]
        title = f"Actual level ({descriptor.unit_label})"
        axis_title = f"Actual level ({descriptor.unit_label})"
        color_title = descriptor.unit_label
        reference_line = None
        colorscale = SEQUENTIAL_SCALE
        color_series = series
        if not series.dropna().empty:
            low = float(series.quantile(0.05))
            high = float(series.quantile(0.95))
            if math.isclose(low, high):
                high = low + 1.0
            range_color = (low, high)
        else:
            range_color = (0.0, 1.0)

    return {
        "value_series": series,
        "color_series": color_series,
        "title": title,
        "axis_title": axis_title,
        "color_title": color_title,
        "reference_line": reference_line,
        "colorscale": colorscale,
        "range_color": range_color,
    }


def build_geo_comparison_copy(
    descriptor,
    display_mode: str,
    window_days: int,
    comparison_label: str,
    spec_title: str,
) -> str:
    if display_mode == "actual_level":
        return f"{spec_title}. Compare To does not change the current-level view."
    return f"{spec_title}. {describe_window_measure(descriptor, window_days).capitalize()} versus {comparison_label}."


def build_region_scope_frame(
    filtered: pd.DataFrame,
    scope: str,
    crop: CropDefinition,
) -> tuple[pd.DataFrame, str]:
    regional = filtered[filtered["geo_level"] == "region"].copy()
    if scope == "all":
        return regional, f"{crop.label} regions"

    if scope in crop.country_codes:
        return regional[regional["country_code"] == scope].copy(), crop.country_lookup[scope].label

    country = crop.region_country_lookup.get(scope)
    if country is None:
        return regional[regional["geo"] == scope].copy(), scope
    return regional[regional["country_code"] == country.code].copy(), country.label


@lru_cache(maxsize=1)
def load_regional_geojson() -> dict[str, object]:
    features: list[dict[str, object]] = []
    boundary_lookup = {name: geo for geo, name in REGION_BOUNDARY_NAME.items()}
    for path in REGIONAL_GEOJSON_FILES.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for feature in payload.get("features", []):
            feature_copy = deepcopy(feature)
            shape_name = str(feature_copy.get("properties", {}).get("shapeName", ""))
            geo_code = boundary_lookup.get(shape_name)
            if geo_code is None:
                continue
            feature_copy["id"] = geo_code
            feature_copy.setdefault("properties", {})["ags_geo"] = geo_code
            features.append(feature_copy)
    return {"type": "FeatureCollection", "features": features}


def build_geo_overview_figure(
    snapshot: pd.DataFrame,
    map_param: str,
    scope: str,
    crop: CropDefinition,
    display_mode: str = "signal_zscore",
    window_days: int = 7,
    comparison_label: str = "same dates last year",
) -> go.Figure:
    descriptor = parse_param_label(map_param)
    filtered = snapshot[snapshot["param"] == map_param].copy()
    filtered["pct_normal"] = safe_percent_of_normal(filtered["window_mean"], filtered["reference_mean"])

    if filtered.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No geo-level data available for this metric.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Geo overview")

    if scope == "all":
        return build_country_map(
            filtered,
            descriptor,
            display_mode=display_mode,
            window_days=window_days,
            comparison_label=comparison_label,
        )

    if scope in crop.country_codes:
        subset = filtered[(filtered["country_code"] == scope) & (filtered["geo_level"] == "region")]
        title_scope = crop.country_lookup[scope].label
    else:
        country_code = filtered.loc[filtered["geo"] == scope, "country_code"]
        country_code = country_code.iloc[0] if not country_code.empty else None
        subset = filtered[(filtered["country_code"] == country_code) & (filtered["geo_level"] == "region")]
        title_scope = filtered.loc[filtered["geo"] == scope, "country_label"].iloc[0] if country_code else scope

    if subset.empty:
        return build_country_map(
            filtered[filtered["geo_level"] == "country"],
            descriptor,
            display_mode=display_mode,
            window_days=window_days,
            comparison_label=comparison_label,
        )

    spec = geo_display_spec(subset, descriptor, display_mode)
    window_measure = describe_window_measure(descriptor, window_days).capitalize()
    subset = subset.assign(
        display_value=spec["value_series"],
        color_value=spec["color_series"],
    ).sort_values(["display_value", "geo_weight"], ascending=[True, False]).copy()
    subset["hover_text"] = (
        subset["geo_label"]
        + "<br>Weight: "
        + subset["geo_weight"].map(lambda value: f"{value:.2f}%")
        + f"<br>{window_measure}: "
        + subset["window_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + f"<br>Reference ({comparison_label}): "
        + subset["reference_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + "<br>% of normal: "
        + subset["pct_normal"].map(lambda value: f"{value:.0f}%" if pd.notna(value) else "n/a")
        + "<br>Anomaly: "
        + subset["anomaly"].map(lambda value: f"{value:+.2f} {descriptor.unit_label}")
        + "<br>Signal: "
        + subset["issue_flag"]
    )
    subset["highlight"] = subset["geo"].eq(scope)

    figure = go.Figure(
        go.Bar(
            x=subset["display_value"],
            y=subset["geo_label"],
            orientation="h",
            marker=dict(
                color=subset["color_value"],
                colorscale=spec["colorscale"],
                cmin=spec["range_color"][0],
                cmax=spec["range_color"][1],
                line=dict(
                    color=["#243740" if flag else "rgba(0,0,0,0)" for flag in subset["highlight"]],
                    width=[2 if flag else 0 for flag in subset["highlight"]],
                ),
                colorbar=dict(title=spec["color_title"]),
            ),
            customdata=subset[["hover_text"]],
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
    )
    if spec["reference_line"] is not None:
        figure.add_vline(
            x=spec["reference_line"],
            line_color="rgba(31, 45, 37, 0.45)",
            line_dash="dash",
        )
    figure.update_yaxes(automargin=True)
    figure.update_xaxes(
        title_text=str(spec["axis_title"]),
        showgrid=True,
        gridcolor=GRID_COLOR,
    )
    return apply_layout(
        figure,
        title=(
            f"{title_scope} | {descriptor.short_label} by subregion"
            f"<br><sup>{build_geo_comparison_copy(descriptor, display_mode, window_days, comparison_label, spec['title'])}</sup>"
        ),
        height=max(420, 36 * len(subset) + 180),
        margin=dict(l=30, r=20, t=90, b=40),
    )


def build_country_map(
    filtered: pd.DataFrame,
    descriptor,
    display_mode: str = "signal_zscore",
    window_days: int = 7,
    comparison_label: str = "same dates last year",
) -> go.Figure:
    if filtered.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No map-ready country data available for this metric.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Palm oil anomaly map")

    filtered = filtered[filtered["geo_level"] == "country"].copy()
    filtered["pct_normal"] = safe_percent_of_normal(filtered["window_mean"], filtered["reference_mean"])
    spec = geo_display_spec(filtered, descriptor, display_mode)
    window_measure = describe_window_measure(descriptor, window_days).capitalize()
    filtered["hover_text"] = (
        filtered["country_label"]
        + "<br>"
        + filtered["metric_label"]
        + f"<br>{window_measure}: "
        + filtered["window_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + f"<br>Reference ({comparison_label}): "
        + filtered["reference_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + "<br>% of normal: "
        + filtered["pct_normal"].map(lambda value: f"{value:.0f}%" if pd.notna(value) else "n/a")
        + "<br>Anomaly: "
        + filtered["anomaly"].map(lambda value: f"{value:+.2f} {descriptor.unit_label}")
        + "<br>Signal: "
        + filtered["issue_flag"]
    )

    figure = px.choropleth(
        filtered,
        locations="iso_alpha3",
        color=spec["color_series"],
        hover_name="country_label",
        custom_data=["hover_text"],
        color_continuous_scale=spec["colorscale"],
        range_color=spec["range_color"],
    )
    figure.update_traces(
        marker_line_color="#27392f",
        marker_line_width=0.8,
        hovertemplate="%{customdata[0]}<extra></extra>",
        colorbar_title=spec["color_title"],
    )
    figure.update_geos(
        fitbounds="locations",
        showcountries=True,
        countrycolor="rgba(39, 57, 47, 0.35)",
        showcoastlines=True,
        coastlinecolor="rgba(39, 57, 47, 0.35)",
        showland=True,
        landcolor="#f1ece3",
        bgcolor="rgba(0,0,0,0)",
    )
    return apply_layout(
        figure,
        title=(
            f"{descriptor.short_label} country map"
            f"<br><sup>{build_geo_comparison_copy(descriptor, display_mode, window_days, comparison_label, spec['title'])}</sup>"
        ),
        height=430,
        margin=dict(l=10, r=10, t=70, b=10),
    )


def build_geo_map_figure(
    snapshot: pd.DataFrame,
    map_param: str,
    scope: str,
    crop: CropDefinition,
    display_mode: str = "signal_zscore",
    window_days: int = 7,
    comparison_label: str = "same dates last year",
) -> go.Figure:
    descriptor = parse_param_label(map_param)
    filtered = snapshot[snapshot["param"] == map_param].copy()
    filtered["pct_normal"] = safe_percent_of_normal(filtered["window_mean"], filtered["reference_mean"])
    subset, title_scope = build_region_scope_frame(filtered, scope, crop)

    if subset.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No region-level data available for this metric.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Regional map")

    subset = subset.copy()
    subset["lat"] = subset["geo"].map(lambda geo: REGION_COORDINATES.get(geo, (math.nan, math.nan))[0])
    subset["lon"] = subset["geo"].map(lambda geo: REGION_COORDINATES.get(geo, (math.nan, math.nan))[1])
    subset = subset.dropna(subset=["lat", "lon"]).copy()
    if subset.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="Region coordinates are not available for this selection.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Regional map")

    spec = geo_display_spec(subset, descriptor, display_mode)
    window_measure = describe_window_measure(descriptor, window_days).capitalize()
    weight_span = max(float(subset["geo_weight"].max()) - float(subset["geo_weight"].min()), 0.01)
    subset["color_value"] = spec["color_series"]
    subset["marker_size"] = 15 + ((subset["geo_weight"] - subset["geo_weight"].min()) / weight_span) * 18
    subset["hover_text"] = (
        subset["geo_label"]
        + "<br>Weight: "
        + subset["geo_weight"].map(lambda value: f"{value:.2f}%")
        + f"<br>{window_measure}: "
        + subset["window_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + f"<br>Reference ({comparison_label}): "
        + subset["reference_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}" if pd.notna(value) else "n/a")
        + "<br>% of normal: "
        + subset["pct_normal"].map(lambda value: f"{value:.0f}%" if pd.notna(value) else "n/a")
        + "<br>Anomaly: "
        + subset["anomaly"].map(lambda value: f"{value:+.2f} {descriptor.unit_label}" if pd.notna(value) else "n/a")
        + "<br>Signal: "
        + subset["issue_flag"]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scattergeo(
            lon=subset["lon"],
            lat=subset["lat"],
            text=subset["geo_label"],
            customdata=subset[["hover_text"]],
            hovertemplate="%{customdata[0]}<extra></extra>",
            mode="markers",
            marker=dict(
                size=subset["marker_size"],
                color=spec["color_series"],
                colorscale=spec["colorscale"],
                cmin=spec["range_color"][0],
                cmax=spec["range_color"][1],
                colorbar=dict(title=spec["color_title"]),
                line=dict(color="rgba(36, 55, 64, 0.55)", width=1.2),
                opacity=0.95,
            ),
            showlegend=False,
        )
    )

    if scope not in {"all", *crop.country_codes} and scope in set(subset["geo"]):
        selected = subset[subset["geo"] == scope].iloc[0]
        figure.add_trace(
            go.Scattergeo(
                lon=[selected["lon"]],
                lat=[selected["lat"]],
                customdata=[[selected["hover_text"]]],
                hovertemplate="%{customdata[0]}<extra></extra>",
                mode="markers",
                marker=dict(
                    size=[float(selected["marker_size"]) + 8],
                    color=[selected["color_value"] if "color_value" in selected else selected["zscore"]],
                    colorscale=spec["colorscale"],
                    cmin=spec["range_color"][0],
                    cmax=spec["range_color"][1],
                    line=dict(color="#243740", width=3),
                    opacity=1.0,
                    showscale=False,
                ),
                showlegend=False,
            )
        )

    lat_pad = max(1.4, (float(subset["lat"].max()) - float(subset["lat"].min())) * 0.28)
    lon_pad = max(1.8, (float(subset["lon"].max()) - float(subset["lon"].min())) * 0.20)
    figure.update_geos(
        projection_type="mercator",
        showcountries=True,
        countrycolor="rgba(39, 57, 47, 0.35)",
        showcoastlines=True,
        coastlinecolor="rgba(39, 57, 47, 0.35)",
        showland=True,
        landcolor="#f1ece3",
        showocean=True,
        oceancolor="#edf4f5",
        bgcolor="rgba(0,0,0,0)",
        lataxis_range=[float(subset["lat"].min()) - lat_pad, float(subset["lat"].max()) + lat_pad],
        lonaxis_range=[float(subset["lon"].min()) - lon_pad, float(subset["lon"].max()) + lon_pad],
    )
    return apply_layout(
        figure,
        title=(
            f"{title_scope} | {descriptor.short_label} regional map"
            f"<br><sup>{build_geo_comparison_copy(descriptor, display_mode, window_days, comparison_label, spec['title'])}</sup>"
        ),
        height=520,
        margin=dict(l=10, r=10, t=90, b=18),
    )


def build_geo_ranking_figure(
    snapshot: pd.DataFrame,
    map_param: str,
    scope: str,
    crop: CropDefinition,
    display_mode: str = "signal_zscore",
    window_days: int = 7,
    comparison_label: str = "same dates last year",
) -> go.Figure:
    return build_geo_overview_figure(
        snapshot=snapshot,
        map_param=map_param,
        scope=scope,
        crop=crop,
        display_mode=display_mode,
        window_days=window_days,
        comparison_label=comparison_label,
    )


def build_recent_context_figure(
    recent_context: pd.DataFrame,
    current_date: pd.Timestamp,
    scope_label: str,
    comparison_label: str = "same dates last year",
    comparison_mode: str = "same_period_last_year",
) -> go.Figure:
    if recent_context.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No history/forecast data available for the selected scope.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="History, current cut, and forecast")

    param_order = sorted(
        recent_context["param"].unique(),
        key=lambda value: STATISTIC_ORDER.get(parse_param_label(value).statistic_code, 99),
    )
    subplot_titles = [parse_param_label(param).ui_metric_label for param in param_order]
    figure = make_subplots(
        rows=len(param_order),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.03, 0.12 / max(len(param_order), 1)),
        subplot_titles=tuple(subplot_titles),
    )

    for row_number, param in enumerate(param_order, start=1):
        subset = recent_context[recent_context["param"] == param].copy()
        subset = subset.sort_values("date")
        actual = subset[subset["date"] <= current_date]
        forecast = subset[subset["date"] > current_date]
        descriptor = parse_param_label(param)
        has_reference_band = subset["reference_low"].notna().any() and subset["reference_high"].notna().any()

        if has_reference_band:
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["reference_high"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_number,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["reference_low"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=BAND_COLOR,
                    line=dict(width=0),
                    name="Reference range" if row_number == 1 else None,
                    showlegend=row_number == 1,
                    hoverinfo="skip",
                ),
                row=row_number,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["reference_mean"],
                mode="lines",
                name="Reference" if row_number == 1 else None,
                showlegend=row_number == 1,
                line=dict(
                    color="rgba(52, 66, 58, 0.70)",
                    width=3,
                    dash="dash" if comparison_mode == "previous_window" else "solid",
                ),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    f"Reference ({comparison_label}): "
                    "%{y:.2f} "
                    f"{descriptor.unit_label}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=actual["date"],
                y=actual["value"],
                mode="lines+markers",
                name="Actual" if row_number == 1 else None,
                showlegend=row_number == 1,
                line=dict(color=ACCENT_COLOR, width=2),
                marker=dict(size=5, color=ACCENT_COLOR),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    f"{descriptor.short_label}: "
                    "%{y:.2f} "
                    f"{descriptor.unit_label}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["value"],
                mode="lines+markers",
                name="Forecast" if row_number == 1 else None,
                showlegend=row_number == 1,
                line=dict(color=FORECAST_COLOR, width=2, dash="dot"),
                marker=dict(size=5, color=FORECAST_COLOR),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    f"{descriptor.short_label}: "
                    "%{y:.2f} "
                    f"{descriptor.unit_label}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )
        figure.update_yaxes(title_text=descriptor.unit_label, row=row_number, col=1)

    figure.add_vline(
        x=current_date,
        line_dash="dash",
        line_color="rgba(31, 45, 37, 0.65)",
        line_width=1.5,
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    if comparison_mode == "previous_window":
        subtitle = (
            f"Reference line shows the {comparison_label} benchmark on the same daily scale; "
            "the vertical dashed line marks the latest release date."
        )
    elif comparison_mode == "same_period_last_year":
        subtitle = (
            "Reference line shows the same dates last year, while the grey band shows the broader historical 10th-90th range; "
            "the vertical dashed line marks the latest release date."
        )
    else:
        subtitle = (
            "Reference line shows the historical normal and the grey band shows the historical 10th-90th range; "
            "the vertical dashed line marks the latest release date."
        )
    return apply_layout(
        figure,
        title=(
            f"{scope_label} | recent path and 15-day forecast"
            f"<br><sup>{subtitle}</sup>"
        ),
        height=max(500, 320 * len(param_order) + 100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=110, b=40),
    )


def build_monthly_heatmap(matrix: pd.DataFrame, scope_label: str) -> go.Figure:
    if matrix.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No monthly issue grid is available for this selection.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Monthly issue grid")

    pivot_z = matrix.pivot(index="row_label", columns="month_label", values="zscore")
    pivot_text = matrix.pivot(index="row_label", columns="month_label", values="display_value")

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=pivot_z.to_numpy(),
                x=list(pivot_z.columns),
                y=list(pivot_z.index),
                text=pivot_text.to_numpy(),
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#2d6c8d"],
                    [0.48, "#f7f5f1"],
                    [0.52, "#f7f5f1"],
                    [1.0, "#cf5d2e"],
                ],
                zmid=0,
                zmin=-2.5,
                zmax=2.5,
                colorbar=dict(title="z-score"),
                hovertemplate=(
                    "%{y}<br>%{x}<br>Monthly anomaly: %{text}"
                    "<br>Signal strength: %{z:.2f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_xaxes(side="top")
    figure.update_yaxes(automargin=True)
    return apply_layout(
        figure,
        title=(
            f"{scope_label} | monthly issue grid"
            "<br><sup>Cells show monthly mean anomalies against the same calendar month in prior years.</sup>"
        ),
        height=max(420, 32 * len(pivot_z.index) + 160),
        margin=dict(l=10, r=10, t=90, b=20),
    )


def build_rainfall_threshold_figure(matrix: pd.DataFrame, scope_label: str) -> go.Figure:
    if matrix.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No complete monthly rainfall totals are available for this geo selection.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Monthly rainfall thresholds")

    rows = list(dict.fromkeys(matrix["row_label"]))
    months = list(dict.fromkeys(matrix["month_start"]))
    month_labels = [month_value.strftime("%b\n%Y") for month_value in months]
    pivot_bucket = (
        matrix.pivot(index="row_label", columns="month_start", values="bucket")
        .reindex(index=rows, columns=months)
    )
    pivot_hover = (
        matrix.pivot(index="row_label", columns="month_start", values="hover_value")
        .reindex(index=rows, columns=months)
    )

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=pivot_bucket.to_numpy(),
                x=months,
                y=rows,
                customdata=pivot_hover.to_numpy(),
                colorscale=[
                    [0.0, "#d83b2d"],
                    [0.32, "#d83b2d"],
                    [0.33, "#eadfcd"],
                    [0.66, "#eadfcd"],
                    [0.67, "#2ba7df"],
                    [1.0, "#2ba7df"],
                ],
                zmin=-1,
                zmax=1,
                showscale=False,
                xgap=3,
                ygap=3,
                hovertemplate="%{y}<br>%{x|%b %Y}<br>%{customdata}<extra></extra>",
            )
        ]
    )
    figure.update_xaxes(
        side="top",
        tickmode="array",
        tickvals=months,
        ticktext=month_labels,
        tickangle=0,
        showgrid=False,
        tickfont=dict(size=10),
    )
    figure.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    return apply_layout(
        figure,
        title=(
            f"{scope_label} | monthly rainfall thresholds"
            "<br><sup>Each cell is one calendar month. Red = under 100 mm/month, blue = over 400 mm/month.</sup>"
        ),
        height=max(340, 40 * len(rows) + 170),
        margin=dict(l=18, r=18, t=95, b=24),
    )


def apply_layout(figure: go.Figure, title: str, **layout_overrides: object) -> go.Figure:
    base_layout = dict(
        title=dict(text=title, x=0.01),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PLOT_COLOR,
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT_COLOR, size=13),
        margin=dict(l=20, r=20, t=70, b=20),
        hoverlabel=dict(bgcolor="#fffaf3", font_color=TEXT_COLOR),
    )
    base_layout.update(layout_overrides)
    figure.update_layout(**base_layout)
    return figure
