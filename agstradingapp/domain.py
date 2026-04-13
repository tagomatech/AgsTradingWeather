from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


CROP_LABELS = {
    "palmoil": "Palm Oil",
}

VARIABLE_LABELS = {
    "t2m": "2m air temperature",
}

VARIABLE_SHORT_LABELS = {
    "t2m": "Temperature",
}

SIGNAL_FAMILY = {
    "t2m": "temperature",
}

STATISTIC_LABELS = {
    "mean": "Average",
    "max": "Maximum",
    "min": "Minimum",
    "sum": "Accumulated",
}

STATISTIC_ORDER = {
    "mean": 0,
    "max": 1,
    "min": 2,
    "sum": 3,
}

UNIT_LABELS = {
    "degree_c": "deg C",
    "mm": "mm",
}


@dataclass(frozen=True)
class ParamDescriptor:
    raw_param: str
    crop_key: str
    crop_label: str
    variable_code: str
    variable_label: str
    signal_family: str
    statistic_code: str
    statistic_label: str
    unit_code: str
    unit_label: str
    ui_metric_label: str
    short_label: str


def _humanize_slug(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def parse_param_label(param: str) -> ParamDescriptor:
    """Parse labels such as palmoil-t2m_mean-degree_c."""
    try:
        crop_key, signal_token, unit_code = param.split("-", 2)
        variable_code, statistic_code = signal_token.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(
            f"Expected a label shaped like crop-variable_stat-unit, got '{param}'."
        ) from exc

    crop_label = CROP_LABELS.get(crop_key, _humanize_slug(crop_key))
    variable_label = VARIABLE_LABELS.get(variable_code, _humanize_slug(variable_code))
    statistic_label = STATISTIC_LABELS.get(statistic_code, _humanize_slug(statistic_code))
    unit_label = UNIT_LABELS.get(unit_code, _humanize_slug(unit_code))
    signal_family = SIGNAL_FAMILY.get(variable_code, "other")
    short_variable = VARIABLE_SHORT_LABELS.get(variable_code, variable_label)

    return ParamDescriptor(
        raw_param=param,
        crop_key=crop_key,
        crop_label=crop_label,
        variable_code=variable_code,
        variable_label=variable_label,
        signal_family=signal_family,
        statistic_code=statistic_code,
        statistic_label=statistic_label,
        unit_code=unit_code,
        unit_label=unit_label,
        ui_metric_label=f"{statistic_label} {variable_label}",
        short_label=f"{statistic_label} {short_variable}",
    )


def build_param_dictionary(params: Iterable[str]) -> pd.DataFrame:
    records = [asdict(parse_param_label(param)) for param in sorted(set(params))]
    glossary = pd.DataFrame.from_records(records)
    if glossary.empty:
        return glossary

    glossary["stat_order"] = glossary["statistic_code"].map(STATISTIC_ORDER).fillna(99)
    glossary = glossary.sort_values(
        ["crop_label", "stat_order", "ui_metric_label", "raw_param"]
    ).reset_index(drop=True)
    return glossary.drop(columns=["stat_order"])
