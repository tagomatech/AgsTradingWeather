from __future__ import annotations

from dash import Dash, Input, Output, dash_table, dcc, html

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
    build_country_map,
    build_monthly_heatmap,
    build_recent_context_figure,
)


DATASET = load_dataset(PALM_OIL)
SNAPSHOT = build_snapshot(DATASET.country_daily, DATASET.current_date)

MAP_OPTIONS = [
    {"label": row["short_label"], "value": row["raw_param"]}
    for _, row in DATASET.param_dictionary.iterrows()
]

SCOPE_OPTIONS = [{"label": "Palm oil core belt", "value": "all"}] + [
    {"label": country.label, "value": country.code}
    for country in PALM_OIL.countries
]

PARAM_GLOSSARY_COLUMNS = [
    {"name": "Raw param", "id": "raw_param"},
    {"name": "UI metric", "id": "ui_metric_label"},
    {"name": "Signal family", "id": "signal_family"},
    {"name": "Statistic", "id": "statistic_label"},
    {"name": "Unit", "id": "unit_label"},
]

ISSUE_COLUMNS = [
    {"name": "Country", "id": "country"},
    {"name": "Metric", "id": "metric"},
    {"name": "Latest", "id": "latest"},
    {"name": "7d mean", "id": "trailing_7d_mean"},
    {"name": "Climatology", "id": "climatology_mean"},
    {"name": "Anomaly", "id": "anomaly"},
    {"name": "z-score", "id": "zscore"},
    {"name": "Signal", "id": "issue_flag"},
]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "agstradingapp"


def render_kpi_cards(summary_cards: list[dict[str, str]]) -> list[html.Div]:
    cards: list[html.Div] = []
    for card in summary_cards:
        cards.append(
            html.Div(
                className="kpi-card",
                children=[
                    html.Div(card["label"], className="kpi-label"),
                    html.Div(card["value"], className="kpi-value"),
                    html.Div(card["detail"], className="kpi-detail"),
                ],
            )
        )
    return cards


def build_issue_table_rows(scope: str) -> list[dict[str, str]]:
    scoped = filter_snapshot(SNAPSHOT, scope).sort_values(
        ["abs_zscore", "country_label"], ascending=[False, True]
    )
    rows: list[dict[str, str]] = []
    for _, row in scoped.iterrows():
        rows.append(
            {
                "country": row["country_label"],
                "metric": row["metric_label"],
                "latest": f"{row['latest_value']:.2f} {row['unit_label']}",
                "trailing_7d_mean": f"{row['trailing_7d_mean']:.2f} {row['unit_label']}",
                "climatology_mean": f"{row['climatology_mean']:.2f} {row['unit_label']}",
                "anomaly": f"{row['anomaly']:+.2f} {row['unit_label']}",
                "zscore": f"{row['zscore']:+.2f}",
                "issue_flag": row["issue_flag"],
            }
        )
    return rows


app.layout = html.Div(
    className="page-shell",
    children=[
        html.Div(
            className="hero-shell",
            children=[
                html.Div(
                    className="hero-copy",
                    children=[
                        html.Div("AGSTRADINGAPP", className="eyebrow"),
                        html.H1("agstradingapp | palm oil monitor", className="hero-title"),
                        html.P(
                            "First build for Indonesia and Malaysia with parameter decoding, issue flags, "
                            "map view, history/current/forecast context, and a monthly issue matrix.",
                            className="hero-subtitle",
                        ),
                    ],
                ),
                html.Div(
                    className="hero-meta",
                    children=[
                        html.Div(
                            DATASET.source_mode.upper(),
                            className=f"meta-pill meta-pill--{DATASET.source_mode}",
                        ),
                        html.Div(
                            f"Latest release date: {DATASET.current_date.strftime('%d %b %Y')}",
                            className="meta-text",
                        ),
                        html.Div(DATASET.status_message, className="meta-note"),
                    ],
                ),
            ],
        ),
        dcc.Tabs(
            className="crop-tabs",
            children=[
                dcc.Tab(
                    label="Palm Oil",
                    className="crop-tab",
                    selected_className="crop-tab crop-tab--selected",
                    children=[
                        html.Div(
                            className="toolbar",
                            children=[
                                html.Div(
                                    className="control-block",
                                    children=[
                                        html.Div("Desk scope", className="control-label"),
                                        dcc.Dropdown(
                                            id="scope-filter",
                                            options=SCOPE_OPTIONS,
                                            value="all",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-block",
                                    children=[
                                        html.Div("Map metric", className="control-label"),
                                        dcc.RadioItems(
                                            id="map-param-filter",
                                            options=MAP_OPTIONS,
                                            value=PALM_OIL.default_map_param,
                                            inline=True,
                                            className="metric-radios",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="kpi-grid", className="kpi-grid"),
                        html.Div(
                            className="hero-grid",
                            children=[
                                html.Div(
                                    className="panel",
                                    children=[dcc.Graph(id="country-map", config={"displayModeBar": False})],
                                ),
                                html.Div(
                                    className="panel",
                                    children=[
                                        html.Div("Param field decoding", className="panel-title"),
                                        html.P(
                                            "Current labels follow the pattern crop-variable_stat-unit. "
                                            "For example, palmoil-t2m_max-degree_c means palm oil, "
                                            "2m air temperature, maximum statistic, in Celsius.",
                                            className="panel-copy",
                                        ),
                                        dash_table.DataTable(
                                            columns=PARAM_GLOSSARY_COLUMNS,
                                            data=DATASET.param_dictionary.to_dict("records"),
                                            page_size=6,
                                            style_as_list_view=True,
                                            style_table={"overflowX": "auto"},
                                            style_header={"fontWeight": "600", "backgroundColor": "#f4ede2"},
                                            style_cell={
                                                "backgroundColor": "rgba(255,255,255,0.85)",
                                                "color": "#1f2d25",
                                                "border": "none",
                                                "fontFamily": "IBM Plex Sans, Segoe UI, sans-serif",
                                                "padding": "10px",
                                                "textAlign": "left",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
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
                            className="dual-grid",
                            children=[
                                html.Div(
                                    className="panel",
                                    children=[
                                        dcc.Graph(
                                            id="monthly-heatmap",
                                            config={"displayModeBar": False},
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="panel",
                                    children=[
                                        html.Div("Current issue table", className="panel-title"),
                                        html.P(
                                            "Signals are ranked by absolute z-score against the same seasonal window "
                                            "in prior years.",
                                            className="panel-copy",
                                        ),
                                        dash_table.DataTable(
                                            id="issues-table",
                                            columns=ISSUE_COLUMNS,
                                            data=[],
                                            page_size=8,
                                            sort_action="native",
                                            style_as_list_view=True,
                                            style_table={"overflowX": "auto"},
                                            style_header={
                                                "fontWeight": "600",
                                                "backgroundColor": "#f4ede2",
                                            },
                                            style_cell={
                                                "backgroundColor": "rgba(255,255,255,0.85)",
                                                "color": "#1f2d25",
                                                "border": "none",
                                                "fontFamily": "IBM Plex Sans, Segoe UI, sans-serif",
                                                "padding": "10px",
                                                "textAlign": "left",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
)


@app.callback(
    Output("kpi-grid", "children"),
    Output("country-map", "figure"),
    Output("recent-context-chart", "figure"),
    Output("monthly-heatmap", "figure"),
    Output("issues-table", "data"),
    Input("scope-filter", "value"),
    Input("map-param-filter", "value"),
)
def refresh_dashboard(scope: str, map_param: str):
    summary_cards = build_kpi_summary(
        snapshot=SNAPSHOT,
        crop=PALM_OIL,
        scope=scope,
        current_date=DATASET.current_date,
    )
    recent_context, scope_label = build_recent_context(
        country_daily=DATASET.country_daily,
        crop=PALM_OIL,
        scope=scope,
        current_date=DATASET.current_date,
    )
    monthly_matrix = build_monthly_issue_matrix(
        country_daily=DATASET.country_daily,
        crop=PALM_OIL,
        scope=scope,
        current_date=DATASET.current_date,
    )

    return (
        render_kpi_cards(summary_cards),
        build_country_map(SNAPSHOT, map_param),
        build_recent_context_figure(recent_context, DATASET.current_date, scope_label),
        build_monthly_heatmap(monthly_matrix, scope_label),
        build_issue_table_rows(scope),
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8052)
