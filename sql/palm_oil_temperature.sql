SELECT
    date,
    date_release,
    year_market,
    param,
    geo,
    value
FROM source.wth
WHERE (
        (param = 'palmoil-t2m_max-degree_c' AND (geo LIKE 'idn%' OR geo LIKE 'mys%'))
     OR (param = 'palmoil-t2m_mean-degree_c' AND (geo LIKE 'idn%' OR geo LIKE 'mys%'))
     OR (param = 'palmoil-t2m_min-degree_c' AND (geo LIKE 'idn%' OR geo LIKE 'mys%'))
)
  AND date >= '2005-01-01'
  AND geo NOT LIKE '%-%'
ORDER BY date DESC, date_release DESC, param, geo;
