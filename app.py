from __future__ import annotations

import os
import socket
import threading
import time

import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html

from agstradingapp.analytics import (
    build_kpi_summary,
    build_monthly_issue_matrix,
    build_recent_context,
    build_snapshot,
    filter_snapshot,
)
from agstradingapp.config import PALM_OIL
from agstradingapp.data import load_dataset
from agstradingapp.figures import (
    build_geo_overview_figure,
    build_monthly_heatmap,
    build_recent_context_figure,
)


def load_app_state():
    try:
        dataset = load_dataset(PALM_OIL)
    except FileNotFoundError as exc:
        return None, pd.DataFrame(), str(exc)
    return dataset, build_snapshot(dataset.geo_daily, dataset.current_date), None


DATASET, SNAPSHOT, FEED_ERROR = load_app_state()

METRIC_OPTIONS = (
    [{"label": row["short_label"], "value": row["raw_param"]} for _, row in DATASET.param_dictionary.iterrows()]
    if DATASET is not None
    else []
)
COUNTRY_OPTIONS = [{"label": "Core belt", "value": "all"}] + [
    {"label": country.label, "value": country.code} for country in PALM_OIL.countries
]

PARAM_GLOSSARY_COLUMNS = [
    {"name": "Raw param", "id": "raw_param"},
    {"name": "UI metric", "id": "ui_metric_label"},
    {"name": "Signal family", "id": "signal_family"},
    {"name": "Statistic", "id": "statistic_label"},
    {"name": "Unit", "id": "unit_label"},
]

ISSUE_COLUMNS = [
    {"name": "Signal", "id": "signal"},
    {"name": "Geo", "id": "geo"},
    {"name": "Metric", "id": "metric"},
    {"name": "Latest", "id": "latest"},
    {"name": "Anomaly", "id": "anomaly"},
    {"name": "z-score", "id": "zscore"},
    {"name": "Weight", "id": "weight"},
]

TABLE_HEADER_STYLE = {"fontWeight": "600", "backgroundColor": "#f4ede2"}
TABLE_CELL_STYLE = {
    "backgroundColor": "rgba(255,255,255,0.88)",
    "color": "#1f2d25",
    "border": "none",
    "fontFamily": "IBM Plex Sans, Segoe UI, sans-serif",
    "padding": "10px 12px",
    "textAlign": "left",
    "whiteSpace": "normal",
    "overflowWrap": "anywhere",
    "height": "auto",
}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Palm Oil Weather Desk"

DASH_HOST = os.getenv("AGSTRADINGAPP_HOST", "127.0.0.1")
DASH_PORT = int(os.getenv("AGSTRADINGAPP_PORT", "8052"))


def build_scope_options(country_focus: str) -> list[dict[str, str]]:
    if country_focus == "all":
        return [{"label": "Core belt overview", "value": "all"}]

    country = PALM_OIL.country_lookup[country_focus]
    options = [{"label": f"{country.label} aggregate", "value": country.code}]
    options.extend({"label": region.label, "value": region.geo} for region in country.regions)
    return options


def render_kpi_cards(summary_cards: list[dict[str, str]]) -> list[html.Div]:
    return [
        html.Div(
            className="kpi-card",
            children=[
                html.Div(card["label"], className="kpi-label"),
                html.Div(card["value"], className="kpi-value"),
                html.Div(card["detail"], className="kpi-detail"),
            ],
        )
        for card in summary_cards
    ]


def build_issue_table_rows(
    scope: str,
    focus_param: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, dict[str, str]]]]:
    scoped = filter_snapshot(SNAPSHOT, scope, PALM_OIL).sort_values(
        ["level_order", "abs_zscore", "geo_weight"],
        ascending=[True, False, False],
    )
    if focus_param:
        scoped = scoped[scoped["param"] == focus_param].copy()

    rows: list[dict[str, str]] = []
    tooltips: list[dict[str, dict[str, str]]] = []
    for _, row in scoped.iterrows():
        rows.append(
            {
                "signal": row["issue_flag"],
                "geo": row["geo_label"],
                "metric": row["metric_label"],
                "latest": f"{row['latest_value']:.2f} {row['unit_label']}",
                "anomaly": f"{row['anomaly']:+.2f} {row['unit_label']}",
                "zscore": f"{row['zscore']:+.2f}",
                "weight": (
                    "--" if row["geo_level"] == "country" else f"{row['geo_weight']:.2f}%"
                ),
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
                "latest": {
                    "value": (
                        f"7d mean: {row['trailing_7d_mean']:.2f} {row['unit_label']}\n"
                        f"Seasonal mean: {row['climatology_mean']:.2f} {row['unit_label']}"
                    ),
                    "type": "text",
                },
            }
        )
    return rows, tooltips


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
    app.run(
        debug=debug,
        host=DASH_HOST,
        port=DASH_PORT,
        use_reloader=False,
    )


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

    st.set_page_config(page_title="Palm Oil Weather Desk", layout="wide")
    st.title("Palm Oil Weather Desk")
    st.caption("Indonesia and Malaysia weather cuts and risk view")

    @st.cache_resource(show_spinner=False)
    def ensure_dash_server() -> bool:
        if is_dash_server_reachable():
            return True

        debug = os.getenv("AGSTRADINGAPP_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
        thread = threading.Thread(
            target=run_dash_server,
            kwargs={"debug": debug},
            daemon=True,
            name="agstradingapp-dash-server",
        )
        thread.start()
        return wait_for_dash_server()

    with st.spinner("Starting dashboard..."):
        ready = ensure_dash_server()

    dash_url = f"http://{DASH_HOST}:{DASH_PORT}"
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
                className="hero-shell",
                children=[
                    html.Div(
                        className="hero-copy",
                        children=[
                            html.Div("AGSTRADINGAPP", className="eyebrow"),
                            html.H1("Palm Oil Weather Desk", className="hero-title"),
                            html.P(
                                "Indonesia and Malaysia weather cuts, regional anomalies, and forward risk.",
                                className="hero-subtitle",
                            ),
                        ],
                    ),
                    html.Div(
                        className="hero-meta",
                        children=[
                            html.Div("CSV FEED", className="meta-pill meta-pill--csv"),
                            html.Div("A local CSV is required before the dashboard can render.", className="meta-text"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="panel missing-panel",
                children=[
                    html.Div("Feed Required", className="panel-title"),
                    html.P(
                        "This app now runs only from the palm oil CSV feed.",
                        className="panel-copy",
                    ),
                    html.Pre(message, className="error-block"),
                ],
            ),
        ],
    )


def build_dashboard_layout() -> html.Div:
    assert DATASET is not None

    return html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="hero-shell",
                children=[
                    html.Div(
                        className="hero-copy",
                        children=[
                            html.Div("AGSTRADINGAPP", className="eyebrow"),
                            html.H1("Palm Oil Weather Desk", className="hero-title"),
                            html.P(
                                "Indonesia and Malaysia weather cuts, regional anomalies, and forward risk.",
                                className="hero-subtitle",
                            ),
                        ],
                    ),
                    html.Div(
                        className="hero-meta",
                        children=[
                            html.Div("CSV FEED", className="meta-pill meta-pill--csv"),
                            html.Div(
                                f"Latest release date: {DATASET.current_date.strftime('%d %b %Y')}",
                                className="meta-text",
                            ),
                            html.Div(DATASET.status_message, className="meta-note"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="workspace-shell",
                children=[
                    html.Aside(
                        className="sidebar",
                        children=[
                            html.Div(
                                className="sidebar-card",
                                children=[
                                    html.Div("Desk Focus", className="control-label"),
                                    dcc.RadioItems(
                                        id="country-filter",
                                        options=COUNTRY_OPTIONS,
                                        value="all",
                                        className="filter-radios",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-card",
                                children=[
                                    html.Div("Region Cut", className="control-label"),
                                    dcc.RadioItems(
                                        id="scope-filter",
                                        options=build_scope_options("all"),
                                        value="all",
                                        className="filter-radios filter-radios--dense",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sidebar-card",
                                children=[
                                    html.Div("Metric", className="control-label"),
                                    dcc.RadioItems(
                                        id="metric-filter",
                                        options=METRIC_OPTIONS,
                                        value=PALM_OIL.default_map_param,
                                        className="filter-radios",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="workspace-main",
                        children=[
                            html.Div(id="kpi-grid", className="kpi-grid"),
                            html.Div(
                                className="panel",
                                children=[
                                    dcc.Graph(
                                        id="recent-context-chart",
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                            html.Div(
                                className="overview-grid",
                                children=[
                                    html.Div(
                                        className="panel",
                                        children=[
                                            dcc.Graph(
                                                id="geo-overview",
                                                config={"displayModeBar": False},
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        className="panel",
                                        children=[
                                            dcc.Graph(
                                                id="monthly-heatmap",
                                                config={"displayModeBar": False},
                                            )
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="panel",
                                children=[
                                    html.Div("Current Issue Table", className="panel-title"),
                                    html.P(
                                        "Sorted by absolute z-score for the selected metric and desk focus.",
                                        className="panel-copy",
                                    ),
                                    dash_table.DataTable(
                                        id="issues-table",
                                        columns=ISSUE_COLUMNS,
                                        data=[],
                                        tooltip_data=[],
                                        sort_action="native",
                                        page_action="none",
                                        style_as_list_view=True,
                                        style_header=TABLE_HEADER_STYLE,
                                        style_cell=TABLE_CELL_STYLE,
                                        style_data_conditional=[
                                            {
                                                "if": {"filter_query": "{signal} contains 'stress'"},
                                                "color": "#a2441d",
                                                "fontWeight": "600",
                                            },
                                            {
                                                "if": {"filter_query": "{signal} contains 'watch'"},
                                                "color": "#8a5c13",
                                            },
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="panel",
                                children=[
                                    html.Div("Param Field Guide", className="panel-title"),
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
        Output("kpi-grid", "children"),
        Output("geo-overview", "figure"),
        Output("recent-context-chart", "figure"),
        Output("monthly-heatmap", "figure"),
        Output("issues-table", "data"),
        Output("issues-table", "tooltip_data"),
        Input("scope-filter", "value"),
        Input("metric-filter", "value"),
    )
    def refresh_dashboard(scope: str, metric: str):
        summary_cards = build_kpi_summary(
            snapshot=SNAPSHOT,
            crop=PALM_OIL,
            scope=scope,
            current_date=DATASET.current_date,
            focus_param=metric,
        )
        recent_context, scope_label = build_recent_context(
            country_daily=DATASET.country_daily,
            geo_daily=DATASET.geo_daily,
            crop=PALM_OIL,
            scope=scope,
            current_date=DATASET.current_date,
            focus_param=metric,
        )
        monthly_matrix = build_monthly_issue_matrix(
            country_daily=DATASET.country_daily,
            geo_daily=DATASET.geo_daily,
            crop=PALM_OIL,
            scope=scope,
            current_date=DATASET.current_date,
            focus_param=metric,
        )
        issue_rows, issue_tooltips = build_issue_table_rows(scope, metric)

        return (
            render_kpi_cards(summary_cards),
            build_geo_overview_figure(SNAPSHOT, metric, scope, PALM_OIL),
            build_recent_context_figure(recent_context, DATASET.current_date, scope_label),
            build_monthly_heatmap(monthly_matrix, scope_label),
            issue_rows,
            issue_tooltips,
        )


if __name__ == "__main__":
    if is_running_under_streamlit():
        render_streamlit_shell()
    else:
        debug = os.getenv("AGSTRADINGAPP_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
        use_reloader = debug and threading.current_thread() is threading.main_thread()
        app.run(
            debug=debug,
            host=DASH_HOST,
            port=DASH_PORT,
            use_reloader=use_reloader,
        )
