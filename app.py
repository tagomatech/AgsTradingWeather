from __future__ import annotations

import os
import socket
import threading
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html

from agstradingapp.analytics import (
    build_recent_context,
    build_snapshot,
    describe_comparison_mode,
    filter_snapshot,
)
from agstradingapp.config import PALM_OIL
from agstradingapp.data import load_dataset
from agstradingapp.domain import (
    describe_latest_measure,
    describe_window_column_label,
    describe_window_measure,
    parse_param_label,
)
from agstradingapp.figures import (
    build_geo_map_figure,
    build_geo_ranking_figure,
    build_recent_context_figure,
)


def load_app_state():
    try:
        dataset = load_dataset(PALM_OIL)
    except FileNotFoundError as exc:
        return None, pd.DataFrame(), str(exc)
    return dataset, dataset.snapshot, None


DATASET, SNAPSHOT, FEED_ERROR = load_app_state()

DEFAULT_COUNTRY = PALM_OIL.countries[0].code
DEFAULT_SCOPE = DEFAULT_COUNTRY
DEFAULT_METRIC = PALM_OIL.default_map_param
DEFAULT_GEO_VIEW = "signal_zscore"
DEFAULT_WINDOW = 7
DEFAULT_COMPARE = "same_period_last_year"

COUNTRY_OPTIONS = [{"label": country.label, "value": country.code} for country in PALM_OIL.countries]
METRIC_OPTIONS = (
    [{"label": row["short_label"], "value": row["raw_param"]} for _, row in DATASET.param_dictionary.iterrows()]
    if DATASET is not None
    else []
)
WINDOW_OPTIONS = [
    {"label": "7d", "value": 7},
    {"label": "14d", "value": 14},
    {"label": "30d", "value": 30},
    {"label": "60d", "value": 60},
    {"label": "90d", "value": 90},
]
COMPARE_OPTIONS = [
    {"label": "Previous window", "value": "previous_window"},
    {"label": "Same dates last year", "value": "same_period_last_year"},
    {"label": "Historical normal", "value": "seasonal_normal"},
]
GEO_VIEW_OPTIONS = [
    {"label": "Signal score", "value": "signal_zscore"},
    {"label": "Anomaly", "value": "anomaly"},
    {"label": "% of normal", "value": "pct_normal"},
    {"label": "Current level", "value": "actual_level"},
]

PARAM_GLOSSARY_COLUMNS = [
    {"name": "Raw param", "id": "raw_param"},
    {"name": "UI metric", "id": "ui_metric_label"},
    {"name": "Signal family", "id": "signal_family"},
    {"name": "Statistic", "id": "statistic_label"},
    {"name": "Unit", "id": "unit_label"},
]

TABLE_HEADER_STYLE = {"fontWeight": "700", "backgroundColor": "#edf1ef", "border": "none"}
TABLE_CELL_STYLE = {
    "backgroundColor": "rgba(255,255,255,0.9)",
    "color": "#243740",
    "border": "none",
    "fontFamily": "IBM Plex Sans, Segoe UI, sans-serif",
    "padding": "10px 12px",
    "textAlign": "left",
    "whiteSpace": "normal",
    "overflowWrap": "anywhere",
    "height": "auto",
}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "AGSTRADINGWEATHERAPP | Palm Oil Desk"

DASH_HOST = os.getenv("AGSTRADINGAPP_HOST", "127.0.0.1")
APP_REVISION = str(Path(__file__).stat().st_mtime_ns)


def resolve_dash_port() -> int:
    explicit_port = os.getenv("AGSTRADINGAPP_PORT")
    if explicit_port:
        return int(explicit_port)

    # Under Streamlit, avoid reusing an old embedded Dash process after code changes.
    if "STREAMLIT_SERVER_PORT" in os.environ:
        return 8200 + (int(APP_REVISION[-4:]) % 400)

    return 8052


DASH_PORT = resolve_dash_port()


def build_scope_options(country_focus: str) -> list[dict[str, str]]:
    country = PALM_OIL.country_lookup[country_focus]
    options = [{"label": f"{country.label} total", "value": country.code}]
    options.extend({"label": region.label, "value": region.geo} for region in country.regions)
    return options


def format_value(value: float | None, unit_label: str | None = None, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    number = f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{number} {unit_label}" if unit_label else number


def build_issue_table_columns(metric: str, window_days: int) -> list[dict[str, str]]:
    return [
        {"name": "Signal", "id": "signal"},
        {"name": "Geo", "id": "geo"},
        {"name": describe_window_column_label(metric, window_days).title(), "id": "window_value"},
        {"name": "Latest day", "id": "latest_day"},
        {"name": "Reference", "id": "reference"},
        {"name": "Anomaly", "id": "anomaly"},
        {"name": "z-score", "id": "zscore"},
        {"name": "Weight", "id": "weight"},
    ]


def build_reading_guide(metric: str, window_days: int, comparison_mode: str) -> list[object]:
    descriptor = parse_param_label(metric)
    window_measure = describe_window_measure(descriptor, window_days)
    latest_measure = describe_latest_measure(descriptor)
    comparison_label = describe_comparison_mode(comparison_mode, window_days)

    chart_line_text = (
        f'The "Reference" line in the recent-path chart shows the {comparison_label} benchmark for each day.'
    )
    if comparison_mode == "same_period_last_year":
        chart_line_text += " The grey band still shows the broader historical range when enough history exists."
    elif comparison_mode == "seasonal_normal":
        chart_line_text += " It is computed from all available prior years that match the same calendar dates."
    else:
        chart_line_text += " It is a rolling average of the prior window, so it stays on the same daily scale as the chart."

    return [
        html.Div("How To Read This View", className="panel-title"),
        html.P(
            f'Anomaly and z-score use the selected {window_measure}. '
            f'For {descriptor.short_label}, that is the number used to color the map, rank regions, and sort the issue table.',
            className="panel-copy panel-copy--tight",
        ),
        html.Ul(
            className="guide-list",
            children=[
                html.Li(
                    f'"Latest day" is the {latest_measure}. For accumulated precipitation, that means rainfall on the release date, not the full {window_days}-day total.'
                ),
                html.Li(
                    f'"Reference" in the table is the benchmark behind the anomaly and z-score for that same {window_measure}.'
                ),
                html.Li(chart_line_text),
                html.Li(
                    "Historical normal uses whatever prior history is available in the feed for those calendar dates. If the feed starts recently, the normal is based on fewer years or stays blank when no history exists."
                ),
            ],
        ),
    ]


def build_issue_summary(metric: str, window_days: int, comparison_mode: str) -> str:
    window_measure = describe_window_measure(metric, window_days)
    comparison_label = describe_comparison_mode(comparison_mode, window_days)
    return (
        f"Rows are sorted by absolute z-score. "
        f'"Latest day" is the newest daily print, while anomaly and z-score compare the selected {window_measure} versus {comparison_label}.'
    )


def build_issue_table_rows(
    snapshot: pd.DataFrame,
    scope: str,
    focus_param: str | None = None,
    window_days: int = 7,
) -> tuple[list[dict[str, str]], list[dict[str, dict[str, str]]]]:
    scoped = filter_snapshot(snapshot, scope, PALM_OIL).sort_values(
        ["level_order", "abs_zscore", "geo_weight"],
        ascending=[True, False, False],
    )
    if focus_param:
        scoped = scoped[scoped["param"] == focus_param].copy()

    rows: list[dict[str, str]] = []
    tooltips: list[dict[str, dict[str, str]]] = []
    for _, row in scoped.iterrows():
        descriptor = parse_param_label(row["param"])
        window_measure = describe_window_measure(descriptor, window_days)
        latest_measure = describe_latest_measure(descriptor)
        rows.append(
            {
                "signal": row["issue_flag"],
                "geo": row["geo_label"],
                "window_value": format_value(row["window_mean"], row["unit_label"]),
                "latest_day": format_value(row["latest_value"], row["unit_label"]),
                "reference": format_value(row["reference_mean"], row["unit_label"]),
                "anomaly": format_value(row["anomaly"], row["unit_label"], signed=True),
                "zscore": format_value(row["zscore"], signed=True),
                "weight": "--" if row["geo_level"] == "country" else f"{row['geo_weight']:.2f}%",
            }
        )
        tooltips.append(
            {
                "geo": {
                    "value": (
                        f"Level: {'Country' if row['geo_level'] == 'country' else 'Region'}\n"
                        f"Country: {row['country_label']}"
                    ),
                    "type": "text",
                },
                "window_value": {
                    "value": (
                        f"{window_measure.title()}: {format_value(row['window_mean'], row['unit_label'])}\n"
                        f"Built from the selected {window_days}-day window."
                    ),
                    "type": "text",
                },
                "latest_day": {
                    "value": (
                        f"{latest_measure.title()}: {format_value(row['latest_value'], row['unit_label'])}\n"
                        "This is the last daily reading in the current cut."
                    ),
                    "type": "text",
                },
                "reference": {
                    "value": (
                        f"{row['reference_label'].title()}: {format_value(row['reference_mean'], row['unit_label'])}\n"
                        f"Historical seasons available: {row['history_years']}"
                    ),
                    "type": "text",
                },
            }
        )
    return rows, tooltips


@lru_cache(maxsize=160)
def compute_snapshot(window_days: int, comparison_mode: str) -> pd.DataFrame:
    if DATASET is None:
        return pd.DataFrame()
    if int(window_days) == DEFAULT_WINDOW and comparison_mode == DEFAULT_COMPARE:
        return SNAPSHOT
    return build_snapshot(
        DATASET.geo_daily,
        DATASET.current_date,
        window_days=int(window_days),
        comparison_mode=comparison_mode,
    )


@lru_cache(maxsize=256)
def compute_dashboard_view(
    country_focus: str,
    scope: str,
    metric: str,
    geo_view_mode: str,
    window_days: int,
    comparison_mode: str,
):
    if DATASET is None:
        return [], "", [], {}, {}, {}, [], []

    snapshot = compute_snapshot(int(window_days), comparison_mode)
    comparison_label = describe_comparison_mode(comparison_mode, int(window_days))

    recent_context, scope_label = build_recent_context(
        country_daily=DATASET.country_daily,
        geo_daily=DATASET.geo_daily,
        crop=PALM_OIL,
        scope=scope,
        current_date=DATASET.current_date,
        core_belt_daily=DATASET.core_belt_daily,
        focus_param=metric,
        window_days=int(window_days),
        comparison_mode=comparison_mode,
    )
    issue_rows, issue_tooltips = build_issue_table_rows(
        snapshot=snapshot,
        scope=scope,
        focus_param=metric,
        window_days=int(window_days),
    )

    return (
        build_reading_guide(metric, int(window_days), comparison_mode),
        build_issue_summary(metric, int(window_days), comparison_mode),
        build_issue_table_columns(metric, int(window_days)),
        build_geo_map_figure(
            snapshot,
            metric,
            scope,
            PALM_OIL,
            display_mode=geo_view_mode,
            window_days=int(window_days),
            comparison_label=comparison_label,
        ),
        build_geo_ranking_figure(
            snapshot,
            metric,
            scope,
            PALM_OIL,
            display_mode=geo_view_mode,
            window_days=int(window_days),
            comparison_label=comparison_label,
        ),
        build_recent_context_figure(
            recent_context,
            DATASET.current_date,
            scope_label,
            comparison_label=comparison_label,
            comparison_mode=comparison_mode,
        ),
        issue_rows,
        issue_tooltips,
    )


DEFAULT_VIEW = (
    compute_dashboard_view(
        DEFAULT_COUNTRY,
        DEFAULT_SCOPE,
        DEFAULT_METRIC,
        DEFAULT_GEO_VIEW,
        DEFAULT_WINDOW,
        DEFAULT_COMPARE,
    )
    if DATASET is not None
    else None
)


def is_running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def is_dash_server_reachable(host: str = DASH_HOST, port: int = DASH_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def run_dash_server(debug: bool) -> None:
    app.run(debug=debug, host=DASH_HOST, port=DASH_PORT, use_reloader=False)


def wait_for_dash_server(host: str = DASH_HOST, port: int = DASH_PORT, timeout: float = 8.0) -> bool:
    start = time.time()
    while time.time() - start <= timeout:
        if is_dash_server_reachable(host=host, port=port):
            return True
        time.sleep(0.2)
    return False


def render_streamlit_shell() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="AGSTRADINGWEATHERAPP", layout="wide")
    st.title("AGSTRADINGWEATHERAPP")
    st.caption("Palm oil weather desk for Indonesia and Malaysia")

    @st.cache_resource(show_spinner=False)
    def ensure_dash_server() -> bool:
        if is_dash_server_reachable():
            return True

        debug = os.getenv("AGSTRADINGAPP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
        thread = threading.Thread(
            target=run_dash_server,
            kwargs={"debug": debug},
            daemon=True,
            name="agstradingweatherapp-dash-server",
        )
        thread.start()
        return wait_for_dash_server()

    with st.spinner("Starting dashboard..."):
        ready = ensure_dash_server()

    dash_url = f"http://{DASH_HOST}:{DASH_PORT}?v={APP_REVISION}"
    if ready:
        st.link_button("Open dashboard in new tab", dash_url)
        components.iframe(dash_url, height=2200, scrolling=True)
    else:
        st.error("The embedded Dash server did not start correctly.")
        st.code(f"python app.py\n# then open {dash_url}", language="bash")


def build_missing_feed_layout(message: str) -> html.Div:
    return html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="header-shell",
                children=[
                    html.Div(
                        className="header-copy",
                        children=[
                            html.Div("AGSTRADINGWEATHERAPP", className="eyebrow"),
                            html.H1("Palm Oil Weather Desk", className="hero-title"),
                            html.P(
                                "Indonesia and Malaysia weather signals and forecast context.",
                                className="hero-subtitle",
                            ),
                        ],
                    ),
                    html.Div(
                        className="status-chips",
                        children=[html.Div("CSV feed required", className="status-chip status-chip--alert")],
                    ),
                ],
            ),
            html.Div(
                className="panel missing-panel",
                children=[
                    html.Div("Feed Required", className="panel-title"),
                    html.P("This app runs only from the palm oil CSV feed.", className="panel-copy"),
                    html.Pre(message, className="error-block"),
                ],
            ),
        ],
    )


def build_dashboard_layout() -> html.Div:
    assert DATASET is not None
    assert DEFAULT_VIEW is not None

    return html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="header-shell",
                children=[
                    html.Div(
                        className="header-copy",
                        children=[
                            html.Div("AGSTRADINGWEATHERAPP", className="eyebrow"),
                            html.H1("Palm Oil Weather Desk", className="hero-title"),
                            html.P(
                                "Compact risk view for Indonesian and Malaysian palm oil weather.",
                                className="hero-subtitle",
                            ),
                        ],
                    ),
                    html.Div(
                        className="status-chips",
                        children=[
                            html.Div("CSV feed", className="status-chip"),
                            html.Div(
                                f"Release {DATASET.current_date.strftime('%d %b %Y')}",
                                className="status-chip status-chip--quiet",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="control-grid",
                children=[
                    html.Div(
                        className="control-card",
                        children=[
                            html.Div("Country", className="toolbar-label"),
                            dcc.RadioItems(
                                id="country-filter",
                                options=COUNTRY_OPTIONS,
                                value=DEFAULT_COUNTRY,
                                className="toolbar-pills toolbar-pills--compact",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-card",
                        children=[
                            html.Div("Area", className="toolbar-label"),
                            dcc.RadioItems(
                                id="scope-filter",
                                options=build_scope_options(DEFAULT_COUNTRY),
                                value=DEFAULT_SCOPE,
                                className="toolbar-pills toolbar-pills--compact toolbar-pills--dense",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-card",
                        children=[
                            html.Div("Metric", className="toolbar-label"),
                            dcc.RadioItems(
                                id="metric-filter",
                                options=METRIC_OPTIONS,
                                value=DEFAULT_METRIC,
                                className="toolbar-pills toolbar-pills--compact toolbar-pills--dense",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-card",
                        children=[
                            html.Div("Show As", className="toolbar-label"),
                            dcc.RadioItems(
                                id="geo-view-filter",
                                options=GEO_VIEW_OPTIONS,
                                value=DEFAULT_GEO_VIEW,
                                className="toolbar-pills toolbar-pills--compact",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-card",
                        children=[
                            html.Div("Signal Window", className="toolbar-label"),
                            dcc.RadioItems(
                                id="window-filter",
                                options=WINDOW_OPTIONS,
                                value=DEFAULT_WINDOW,
                                className="toolbar-pills toolbar-pills--compact",
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-card control-card--wide",
                        children=[
                            html.Div("Compare To", className="toolbar-label"),
                            dcc.RadioItems(
                                id="comparison-filter",
                                options=COMPARE_OPTIONS,
                                value=DEFAULT_COMPARE,
                                className="toolbar-pills",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="panel panel--guide",
                children=[html.Div(id="reading-guide", children=DEFAULT_VIEW[0])],
            ),
            html.Div(
                className="viz-grid",
                children=[
                    html.Div(
                        className="panel",
                        children=[
                            dcc.Graph(
                                id="geo-map",
                                figure=DEFAULT_VIEW[3],
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            dcc.Graph(
                                id="geo-ranking",
                                figure=DEFAULT_VIEW[4],
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                className="dashboard-stack",
                children=[
                    html.Div(
                        className="panel panel--recent-context",
                        children=[
                            dcc.Graph(
                                id="recent-context-chart",
                                figure=DEFAULT_VIEW[5],
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Div("Current Issue Table", className="panel-title"),
                            html.P(DEFAULT_VIEW[1], className="panel-copy", id="issues-copy"),
                            html.Div(
                                className="signal-threshold-note",
                                children=[
                                    html.Div("Signal Thresholds", className="signal-threshold-title"),
                                    html.P(
                                        "Signals use z-score cutoffs on the selected window/comparison basis. "
                                        "Precipitation: Wet stress >= +1.8, Wet watch >= +1.0, Dry watch <= -1.0, Dry stress <= -1.8. "
                                        "Temperature: Heat stress >= +1.8, Warm watch >= +1.0, Cool watch <= -1.0, Cool stress <= -1.8. "
                                        "Values between -1.0 and +1.0 are shown as Normal.",
                                        className="panel-copy panel-copy--tight",
                                    ),
                                ],
                            ),
                            dash_table.DataTable(
                                id="issues-table",
                                columns=DEFAULT_VIEW[2],
                                data=DEFAULT_VIEW[6],
                                tooltip_data=DEFAULT_VIEW[7],
                                sort_action="native",
                                page_action="none",
                                style_as_list_view=True,
                                style_header=TABLE_HEADER_STYLE,
                                style_cell=TABLE_CELL_STYLE,
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{signal} contains 'stress'"},
                                        "color": "#b44c3d",
                                        "fontWeight": "700",
                                    },
                                    {
                                        "if": {"filter_query": "{signal} contains 'watch'"},
                                        "color": "#9b7a3c",
                                    },
                                ],
                            ),
                        ],
                    ),
                    html.Details(
                        className="panel details-panel",
                        children=[
                            html.Summary("Param Field Guide", className="details-summary"),
                            html.P(
                                "Labels follow crop-variable_stat-unit, for example palmoil-t2m_max-degree_c.",
                                className="panel-copy",
                            ),
                            dash_table.DataTable(
                                columns=PARAM_GLOSSARY_COLUMNS,
                                data=DATASET.param_dictionary.to_dict("records"),
                                style_as_list_view=True,
                                page_action="none",
                                style_header=TABLE_HEADER_STYLE,
                                style_cell=TABLE_CELL_STYLE,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = build_missing_feed_layout(FEED_ERROR) if FEED_ERROR else build_dashboard_layout()


if DATASET is not None:

    @app.callback(
        Output("scope-filter", "options"),
        Output("scope-filter", "value"),
        Input("country-filter", "value"),
        State("scope-filter", "value"),
    )
    def sync_scope_filter(country_focus: str, current_scope: str | None):
        options = build_scope_options(country_focus)
        valid_values = {option["value"] for option in options}
        next_scope = current_scope if current_scope in valid_values else options[0]["value"]
        return options, next_scope


    @app.callback(
        Output("reading-guide", "children"),
        Output("issues-copy", "children"),
        Output("issues-table", "columns"),
        Output("geo-map", "figure"),
        Output("geo-ranking", "figure"),
        Output("recent-context-chart", "figure"),
        Output("issues-table", "data"),
        Output("issues-table", "tooltip_data"),
        Input("country-filter", "value"),
        Input("scope-filter", "value"),
        Input("metric-filter", "value"),
        Input("geo-view-filter", "value"),
        Input("window-filter", "value"),
        Input("comparison-filter", "value"),
    )
    def refresh_dashboard(
        country_focus: str,
        scope: str,
        metric: str,
        geo_view_mode: str,
        window_days: int,
        comparison_mode: str,
    ):
        return compute_dashboard_view(
            country_focus,
            scope,
            metric,
            geo_view_mode,
            int(window_days),
            comparison_mode,
        )


if __name__ == "__main__":
    if is_running_under_streamlit():
        render_streamlit_shell()
    else:
        debug = os.getenv("AGSTRADINGAPP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
        use_reloader = debug and threading.current_thread() is threading.main_thread()
        app.run(debug=debug, host=DASH_HOST, port=DASH_PORT, use_reloader=use_reloader)
