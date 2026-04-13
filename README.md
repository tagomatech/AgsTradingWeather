# agstradingapp

First pass of a crop weather monitor focused on palm oil in Indonesia and Malaysia.

## What is in this build

- Dash + Plotly web app with a single crop tab for palm oil.
- Param-field decoder for labels such as `palmoil-t2m_mean-degree_c`.
- Country issue map for Indonesia and Malaysia.
- History/current/forecast chart with seasonal reference bands.
- Monthly issue heatmap and ranked issue table for the desk.
- Live-query adapter for `source.wth` with deterministic sample fallback.

## Param label breakdown

Current labels follow:

`crop-variable_stat-unit`

Example:

`palmoil-t2m_max-degree_c`

- `palmoil`: crop namespace.
- `t2m`: 2m air temperature.
- `max`: aggregation/statistic.
- `degree_c`: units in Celsius.

## Running the app

```bash
python app.py
```

The app runs at `http://127.0.0.1:8052` as `agstradingapp`.

## Live database mode

Set:

- `WEATHER_DB_URL`
- Optional: `WEATHER_SQL_TABLE` (defaults to `source.wth`)
- Optional: `WEATHER_START_DATE` (defaults to `2005-01-01`)

If no DB URL is configured, the app uses deterministic sample data shaped like the expected query output so the layout and analytics still render.

## Query reference

The starting palm-oil temperature query is saved in [sql/palm_oil_temperature.sql](sql/palm_oil_temperature.sql).
