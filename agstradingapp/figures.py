from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CropDefinition
from .domain import STATISTIC_ORDER, parse_param_label


PAPER_COLOR = "rgba(250, 246, 239, 0)"
PLOT_COLOR = "rgba(255, 255, 255, 0.92)"
GRID_COLOR = "rgba(80, 101, 92, 0.16)"
TEXT_COLOR = "#1f2d25"
ACCENT_COLOR = "#1e6655"
FORECAST_COLOR = "#cf6b28"
BAND_COLOR = "rgba(38, 55, 46, 0.12)"


def build_geo_overview_figure(
    snapshot: pd.DataFrame,
    map_param: str,
    scope: str,
    crop: CropDefinition,
) -> go.Figure:
    descriptor = parse_param_label(map_param)
    filtered = snapshot[snapshot["param"] == map_param].copy()

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
        return build_country_map(filtered, descriptor)

    if scope in crop.country_codes:
        subset = filtered[(filtered["country_code"] == scope) & (filtered["geo_level"] == "region")]
        title_scope = crop.country_lookup[scope].label
    else:
        country_code = filtered.loc[filtered["geo"] == scope, "country_code"]
        country_code = country_code.iloc[0] if not country_code.empty else None
        subset = filtered[(filtered["country_code"] == country_code) & (filtered["geo_level"] == "region")]
        title_scope = filtered.loc[filtered["geo"] == scope, "country_label"].iloc[0] if country_code else scope

    if subset.empty:
        return build_country_map(filtered[filtered["geo_level"] == "country"], descriptor)

    subset = subset.sort_values(["geo_weight", "anomaly"], ascending=[True, True]).copy()
    subset["hover_text"] = (
        subset["geo_label"]
        + "<br>Weight: "
        + subset["geo_weight"].map(lambda value: f"{value:.2f}%")
        + "<br>7-day mean: "
        + subset["trailing_7d_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + "<br>Anomaly: "
        + subset["anomaly"].map(lambda value: f"{value:+.2f} {descriptor.unit_label}")
        + "<br>Signal: "
        + subset["issue_flag"]
    )
    subset["highlight"] = subset["geo"].eq(scope)

    figure = go.Figure(
        go.Bar(
            x=subset["anomaly"],
            y=subset["geo_label"],
            orientation="h",
            marker=dict(
                color=subset["zscore"],
                colorscale=["#2d6c8d", "#eef2ec", "#cf5d2e"],
                cmin=-2.5,
                cmax=2.5,
                line=dict(
                    color=["#1f2d25" if flag else "rgba(0,0,0,0)" for flag in subset["highlight"]],
                    width=[2 if flag else 0 for flag in subset["highlight"]],
                ),
                colorbar=dict(title="z-score"),
            ),
            customdata=subset[["hover_text"]],
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="rgba(31, 45, 37, 0.45)", line_dash="dash")
    figure.update_yaxes(automargin=True)
    figure.update_xaxes(
        title_text=f"Anomaly ({descriptor.unit_label})",
        showgrid=True,
        gridcolor=GRID_COLOR,
    )
    return apply_layout(
        figure,
        title=(
            f"{title_scope} | {descriptor.short_label} anomaly by subregion"
            "<br><sup>Bars are coloured by z-score and weighted by palm-area focus from the reports.</sup>"
        ),
        height=max(420, 36 * len(subset) + 180),
        margin=dict(l=30, r=20, t=90, b=40),
    )


def build_country_map(filtered: pd.DataFrame, descriptor) -> go.Figure:
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
    filtered["hover_text"] = (
        filtered["country_label"]
        + "<br>"
        + filtered["metric_label"]
        + "<br>7-day mean: "
        + filtered["trailing_7d_mean"].map(lambda value: f"{value:.2f} {descriptor.unit_label}")
        + "<br>Anomaly: "
        + filtered["anomaly"].map(lambda value: f"{value:+.2f} {descriptor.unit_label}")
        + "<br>Signal: "
        + filtered["issue_flag"]
    )

    figure = px.choropleth(
        filtered,
        locations="iso_alpha3",
        color="zscore",
        hover_name="country_label",
        custom_data=["hover_text"],
        color_continuous_scale=["#2d6c8d", "#eef2ec", "#cf5d2e"],
        range_color=(-2.5, 2.5),
    )
    figure.update_traces(
        marker_line_color="#27392f",
        marker_line_width=0.8,
        hovertemplate="%{customdata[0]}<extra></extra>",
        colorbar_title="z-score",
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
            f"{descriptor.short_label} anomaly map"
            "<br><sup>Country aggregates are shown here; choose a country or state focus to inspect subregions.</sup>"
        ),
        height=430,
        margin=dict(l=10, r=10, t=70, b=10),
    )


def build_recent_context_figure(
    recent_context: pd.DataFrame,
    current_date: pd.Timestamp,
    scope_label: str,
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

        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["clim_q90"],
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
                y=subset["clim_q10"],
                mode="lines",
                fill="tonexty",
                fillcolor=BAND_COLOR,
                line=dict(width=0),
                name="10th-90th percentile band" if row_number == 1 else None,
                showlegend=row_number == 1,
                hoverinfo="skip",
            ),
            row=row_number,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["clim_mean"],
                mode="lines",
                name="Seasonal mean" if row_number == 1 else None,
                showlegend=row_number == 1,
                line=dict(color="rgba(52, 66, 58, 0.70)", width=3),
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
    return apply_layout(
        figure,
        title=(
            f"{scope_label} | recent path and 15-day forecast"
            "<br><sup>Grey band shows the 10th-90th range from prior years; the dashed line marks the latest release date.</sup>"
        ),
        height=max(420, 280 * len(param_order) + 80),
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
