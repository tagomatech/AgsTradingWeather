from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .domain import STATISTIC_ORDER, parse_param_label


PAPER_COLOR = "rgba(250, 246, 239, 0)"
PLOT_COLOR = "rgba(255, 255, 255, 0.92)"
GRID_COLOR = "rgba(80, 101, 92, 0.16)"
TEXT_COLOR = "#1f2d25"
ACCENT_COLOR = "#1e6655"
FORECAST_COLOR = "#cf6b28"
BAND_COLOR = "rgba(38, 55, 46, 0.12)"


def build_country_map(snapshot: pd.DataFrame, map_param: str) -> go.Figure:
    filtered = snapshot[snapshot["param"] == map_param].copy()
    descriptor = parse_param_label(map_param)

    if filtered.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No map-ready data available for this metric.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return apply_layout(figure, title="Palm oil temperature map")

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
            f"<br><sup>Trailing 7-day signal versus the same seasonal window in prior years.</sup>"
        ),
        height=430,
        margin=dict(l=10, r=10, t=70, b=10),
    )


def build_recent_context_figure(
    recent_context: pd.DataFrame,
    current_date: pd.Timestamp,
    scope_label: str,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Average 2m air temperature",
            "Maximum 2m air temperature",
            "Minimum 2m air temperature",
        ),
    )

    if recent_context.empty:
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
            f"{scope_label} | history, current cut, and 15-day forecast"
            "<br><sup>Grey band shows the 10th-90th seasonal range; the dashed line marks the latest release date.</sup>"
        ),
        height=920,
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
                    "%{y}<br>%{x}<br>Monthly anomaly: %{text} deg C"
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
            "<br><sup>Cells show monthly mean anomalies against the same calendar month across the historical sample.</sup>"
        ),
        height=520,
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
