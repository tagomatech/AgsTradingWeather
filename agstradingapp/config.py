from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionDefinition:
    geo: str
    label: str
    weight: float


@dataclass(frozen=True)
class CountryDefinition:
    code: str
    label: str
    iso_alpha3: str
    weight: float
    regions: tuple[RegionDefinition, ...]


@dataclass(frozen=True)
class CropDefinition:
    crop_id: str
    label: str
    countries: tuple[CountryDefinition, ...]
    params: tuple[str, ...]
    default_map_param: str
    data_filename: str

    @property
    def country_codes(self) -> tuple[str, ...]:
        return tuple(country.code for country in self.countries)

    @property
    def country_weights(self) -> dict[str, float]:
        return {country.code: country.weight for country in self.countries}

    @property
    def country_lookup(self) -> dict[str, CountryDefinition]:
        return {country.code: country for country in self.countries}

    @property
    def region_lookup(self) -> dict[str, RegionDefinition]:
        lookup: dict[str, RegionDefinition] = {}
        for country in self.countries:
            for region in country.regions:
                lookup[region.geo] = region
        return lookup

    @property
    def region_country_lookup(self) -> dict[str, CountryDefinition]:
        lookup: dict[str, CountryDefinition] = {}
        for country in self.countries:
            for region in country.regions:
                lookup[region.geo] = country
        return lookup

    @property
    def all_geo_codes(self) -> tuple[str, ...]:
        codes: list[str] = []
        for country in self.countries:
            codes.append(country.code)
            codes.extend(region.geo for region in country.regions)
        return tuple(codes)

    @property
    def country_regions(self) -> dict[str, tuple[RegionDefinition, ...]]:
        return {country.code: country.regions for country in self.countries}


PALM_OIL = CropDefinition(
    crop_id="palmoil",
    label="Palm Oil",
    countries=(
        CountryDefinition(
            code="idn",
            label="Indonesia",
            iso_alpha3="IDN",
            weight=0.62,
            regions=(
                RegionDefinition(geo="idn-riau", label="Riau", weight=22.10),
                RegionDefinition(geo="idn-kalimantan_tengah", label="Kalimantan Tengah", weight=13.82),
                RegionDefinition(geo="idn-kalimantan_barat", label="Kalimantan Barat", weight=11.15),
                RegionDefinition(geo="idn-kalimantan_timur", label="Kalimantan Timur", weight=9.44),
                RegionDefinition(geo="idn-sumatera_selatan", label="Sumatera Selatan", weight=8.91),
                RegionDefinition(geo="idn-sumatera_utara", label="Sumatera Utara", weight=7.88),
                RegionDefinition(geo="idn-jambi", label="Jambi", weight=7.03),
                RegionDefinition(geo="idn-kalimantan_selatan", label="Kalimantan Selatan", weight=3.58),
                RegionDefinition(geo="idn-aceh", label="Aceh", weight=2.42),
                RegionDefinition(geo="idn-bangka_belitung", label="Bangka Belitung", weight=2.37),
                RegionDefinition(geo="idn-bengkulu", label="Bengkulu", weight=2.06),
                RegionDefinition(geo="idn-sumatera_barat", label="Sumatera Barat", weight=2.06),
                RegionDefinition(geo="idn-lampung", label="Lampung", weight=1.48),
                RegionDefinition(geo="idn-kalimantan_utara", label="Kalimantan Utara", weight=1.30),
                RegionDefinition(geo="idn-papua", label="Papua", weight=1.16),
            ),
        ),
        CountryDefinition(
            code="mys",
            label="Malaysia",
            iso_alpha3="MYS",
            weight=0.38,
            regions=(
                RegionDefinition(geo="mys-sabah", label="Sabah", weight=25.25),
                RegionDefinition(geo="mys-sarawak", label="Sarawak", weight=22.72),
                RegionDefinition(geo="mys-johor", label="Johor", weight=15.82),
                RegionDefinition(geo="mys-pahang", label="Pahang", weight=14.94),
                RegionDefinition(geo="mys-perak", label="Perak", weight=7.71),
                RegionDefinition(geo="mys-negeri_sembilan", label="Negeri Sembilan", weight=3.21),
                RegionDefinition(geo="mys-selangor", label="Selangor", weight=2.83),
                RegionDefinition(geo="mys-trengganu", label="Trengganu", weight=2.65),
                RegionDefinition(geo="mys-kelantan", label="Kelantan", weight=2.09),
                RegionDefinition(geo="mys-kedah", label="Kedah", weight=1.57),
                RegionDefinition(geo="mys-melaka", label="Melaka", weight=1.01),
            ),
        ),
    ),
    params=(
        "palmoil-t2m_mean-degree_c",
        "palmoil-t2m_max-degree_c",
        "palmoil-t2m_min-degree_c",
    ),
    default_map_param="palmoil-t2m_mean-degree_c",
    data_filename="palm_oil_weather_feed.csv",
)
