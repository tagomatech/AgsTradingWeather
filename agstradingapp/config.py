from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountryDefinition:
    code: str
    label: str
    iso_alpha3: str
    weight: float


@dataclass(frozen=True)
class CropDefinition:
    crop_id: str
    label: str
    countries: tuple[CountryDefinition, ...]
    params: tuple[str, ...]
    default_map_param: str

    @property
    def country_codes(self) -> tuple[str, ...]:
        return tuple(country.code for country in self.countries)

    @property
    def country_weights(self) -> dict[str, float]:
        return {country.code: country.weight for country in self.countries}

    @property
    def country_lookup(self) -> dict[str, CountryDefinition]:
        return {country.code: country for country in self.countries}


PALM_OIL = CropDefinition(
    crop_id="palmoil",
    label="Palm Oil",
    countries=(
        CountryDefinition(code="idn", label="Indonesia", iso_alpha3="IDN", weight=0.62),
        CountryDefinition(code="mys", label="Malaysia", iso_alpha3="MYS", weight=0.38),
    ),
    params=(
        "palmoil-t2m_mean-degree_c",
        "palmoil-t2m_max-degree_c",
        "palmoil-t2m_min-degree_c",
    ),
    default_map_param="palmoil-t2m_mean-degree_c",
)
