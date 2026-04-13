# agstradingapp

First pass of a crop weather monitor focused on palm oil in Indonesia and Malaysia.

## What is in this build

- Dash + Plotly web app with a single crop tab for palm oil.
- Param-field decoder for labels such as `palmoil-t2m_mean-degree_c`.
- Country issue map for Indonesia and Malaysia.
- History/current/forecast chart with seasonal reference bands.
- Monthly issue heatmap and ranked issue table for the desk.
- Local CSV feed support with deterministic sample fallback.

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

## Data

If `data/palm_oil_weather_feed.csv` exists, the app uses it.
Otherwise it falls back to built-in sample data.
