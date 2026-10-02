"""EPW and DDY weather files -> the per-city ``ep.weather_city`` table.

Two sources per city. The **EPW** holds 8,760 hourly rows, reduced here to a
monthly summary and then pivoted wide, so each city becomes one row carrying
every metric for every month (``t_mean_c_01`` … ``cdd26_12``). The **DDY** holds
EnergyPlus design days, reduced to summer and winter aggregates.

A model is joined to its city by stripping the trailing ``_<nnnn>`` from its
``model_id`` and lower-casing the rest. That heuristic agrees with the BEM4AI
manifest for every model in the released set.

Two known limitations live in this module; fixing either would change the
published results (see ``docs/limitations.md``):

* design-day wind statistics are absent for part of the city set and become
  ``0.0`` downstream rather than being marked missing;
* ``summer_humidity_value_mean`` averages design-day humidity across condition
  types (enthalpy and wet-bulb), which are orders of magnitude apart, so the
  column is not comparable between cities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text as sa_text

from beg.db.utils import ensure_btree_index, ensure_standard_relation_indexes

#: EPW data-section column order, from the EPW dictionary.
EPW_DATA_COLUMNS = [
    "year", "month", "day", "hour", "minute", "data_source_flags",
    "dry_bulb_temp_c", "dew_point_temp_c", "rel_humidity_percent", "atm_pressure_pa",
    "etr_horiz_whm2", "etr_norm_whm2", "horiz_ir_whm2",
    "ghi_whm2", "dni_whm2", "dhi_whm2",
    "ghi_illux", "dni_illux", "dhi_illux", "zenith_luminance_cdm2",
    "wind_direction_deg", "wind_speed_ms",
    "total_sky_cover_tenths", "opaque_sky_cover_tenths",
    "visibility_km", "ceiling_height_m",
    "present_weather_obs", "present_weather_codes",
    "precipitable_water_mm", "aerosol_optical_depth",
    "snow_depth_cm", "days_since_last_snow", "albedo",
    "liquid_precip_depth_mm", "liquid_precip_rate_hr",
]

#: Per-month metrics computed from the hourly rows and pivoted into wide columns.
EPW_METRICS = [
    "t_mean_c", "t_min_c", "t_max_c", "t_day_mean_c", "t_night_mean_c",
    "rh_mean_pct", "rh80_hours", "wind_speed_mean_ms", "strong_wind_hours",
    "ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2", "daylight_hours",
    "precip_mm", "precip_hours", "hdd18", "cdd26",
]

#: Location columns carried through from the EPW header.
EPW_BASE_COLS = [
    "file_id", "epw_file_name", "country", "state_province", "source", "wmo",
    "latitude", "longitude", "time_zone", "elevation_m",
]


# --------------------------------------------------------------------------- #
# EPW
# --------------------------------------------------------------------------- #


def parse_epw_header(path: Path) -> dict:
    """Read the ``LOCATION`` line of an EPW into a metadata dict."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    parts = [p.strip() for p in first_line.split(",")]
    if not parts or not parts[0].upper().startswith("LOCATION"):
        raise ValueError(f"{path} does not start with a LOCATION header")

    def at(i):
        try:
            return parts[i]
        except IndexError:
            return None

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "file_name": path.name,
        "city": at(1),
        "state_province": at(2),
        "country": at(3),
        "source": at(4),
        "wmo": at(5),
        "latitude": to_float(at(6)),
        "longitude": to_float(at(7)),
        "time_zone": to_float(at(8)),
        "elevation_m": to_float(at(9)),
    }


def read_epw_data(path: Path) -> pd.DataFrame:
    """Read the 8,760 hourly rows that follow an EPW's eight header lines."""
    return pd.read_csv(path, skiprows=8, header=None, names=EPW_DATA_COLUMNS)


def compute_epw_monthly_summary(hourly_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Reduce hourly rows to one row per month, carrying the location metadata.

    Day and night means split on whether global horizontal irradiance is above
    zero, so "daytime temperature" means daylight rather than a clock window.
    Degree days divide summed hourly differences by 24.
    """
    df = hourly_df.copy()
    numeric_cols = [
        "month", "dry_bulb_temp_c", "rel_humidity_percent",
        "ghi_whm2", "dni_whm2", "dhi_whm2", "wind_speed_ms", "liquid_precip_depth_mm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["t_day"] = df["dry_bulb_temp_c"].where(df["ghi_whm2"] > 0)
    df["t_night"] = df["dry_bulb_temp_c"].where(df["ghi_whm2"] <= 0)
    df["rh80_hours"] = (df["rel_humidity_percent"] >= 80).astype(int)
    df["strong_wind_hours"] = (df["wind_speed_ms"] >= 8).astype(int)
    df["daylight_hours"] = (df["ghi_whm2"] > 120).astype(int)
    df["precip_hours"] = (df["liquid_precip_depth_mm"] > 0).astype(int)
    df["hdd18"] = np.where(df["dry_bulb_temp_c"] < 18, 18 - df["dry_bulb_temp_c"], 0)
    df["cdd26"] = np.where(df["dry_bulb_temp_c"] > 26, df["dry_bulb_temp_c"] - 26, 0)

    summary = (
        df.groupby("month")
        .agg(
            t_mean_c=("dry_bulb_temp_c", "mean"),
            t_min_c=("dry_bulb_temp_c", "min"),
            t_max_c=("dry_bulb_temp_c", "max"),
            t_day_mean_c=("t_day", "mean"),
            t_night_mean_c=("t_night", "mean"),
            rh_mean_pct=("rel_humidity_percent", "mean"),
            rh80_hours=("rh80_hours", "sum"),
            wind_speed_mean_ms=("wind_speed_ms", "mean"),
            strong_wind_hours=("strong_wind_hours", "sum"),
            ghi_kwh_m2=("ghi_whm2", lambda s: s.sum() / 1000.0),
            dni_kwh_m2=("dni_whm2", lambda s: s.sum() / 1000.0),
            dhi_kwh_m2=("dhi_whm2", lambda s: s.sum() / 1000.0),
            daylight_hours=("daylight_hours", "sum"),
            precip_mm=("liquid_precip_depth_mm", "sum"),
            precip_hours=("precip_hours", "sum"),
            hdd18=("hdd18", lambda s: s.sum() / 24.0),
            cdd26=("cdd26", lambda s: s.sum() / 24.0),
        )
        .reset_index()
    )

    summary.insert(0, "file_id", meta["file_id"])
    summary.insert(1, "epw_file_name", meta["file_name"])
    for key in ("country", "state_province", "source", "wmo",
                "latitude", "longitude", "time_zone", "elevation_m"):
        summary[key] = meta.get(key)
    return summary


def load_epw_files(epw_files, keep_hourly: bool = False):
    """Parse every EPW into header metadata and a monthly summary.

    Returns ``(files_df, monthly_df, hourly_df_or_None)``. The hourly frame is
    roughly 8,760 rows per city and is kept only on request.
    """
    files_meta, monthly_parts, hourly_parts = [], [], []

    for file_id, path in enumerate(epw_files, start=1):
        path = Path(path)
        meta = parse_epw_header(path)
        meta["file_id"] = file_id
        files_meta.append(meta)

        hourly = read_epw_data(path)
        hourly["file_id"] = file_id
        hourly["epw_file_name"] = meta["file_name"]

        monthly_parts.append(compute_epw_monthly_summary(hourly, meta))
        if keep_hourly:
            hourly_parts.append(hourly)

    files_df = pd.DataFrame(files_meta)
    monthly_df = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    hourly_df = (
        (pd.concat(hourly_parts, ignore_index=True) if hourly_parts else pd.DataFrame())
        if keep_hourly
        else None
    )
    return files_df, monthly_df, hourly_df


def summarise_epw_by_city(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the monthly summary wide: one row per city, ``<metric>_<MM>`` columns."""
    if monthly_df.empty:
        return pd.DataFrame()

    pivot = monthly_df.pivot_table(
        index=EPW_BASE_COLS, columns="month", values=EPW_METRICS, aggfunc="first"
    )
    pivot.columns = [f"{metric}_{int(month):02d}" for metric, month in pivot.columns]
    return pivot.reset_index()


# --------------------------------------------------------------------------- #
# DDY
# --------------------------------------------------------------------------- #

#: DDY field label -> (column, caster). Several spellings map to one column
#: because EnergyPlus versions label wind speed and wind direction differently.
DDY_FIELD_MAP = {
    "Name": ("design_day_name", str),
    "Month": ("month", int),
    "Day of Month": ("day_of_month", int),
    "Day Type": ("day_type", str),
    "Maximum Dry-Bulb Temperature {C}": ("max_drybulb_c", float),
    "Daily Dry-Bulb Temperature Range {C}": ("daily_range_c", float),
    "Humidity Condition Type": ("humidity_condition_type", str),
    "Barometric Pressure {Pa}": ("barometric_pressure_pa", float),
    "Wind Speed {m/s} design conditions vs. traditional 3.35 m/s (7mph)": ("wind_speed_ms", float),
    "Wind Speed {m/s} design conditions vs. traditional 6.71 m/s (15 mph)": ("wind_speed_ms", float),
    "Wind Speed {m/s}": ("wind_speed_ms", float),
    "Wind Direction {Degrees; N=0, S=180}": ("wind_direction_deg", float),
    "Wind Direction {Degrees}": ("wind_direction_deg", float),
    "Wind Direction {Degrees": ("wind_direction_deg", float),
    "Rain {Yes/No}": ("rain_flag", lambda v: v.lower().startswith("y")),
    "Snow on ground {Yes/No}": ("snow_flag", lambda v: v.lower().startswith("y")),
    "Daylight Savings Time Indicator": ("daylight_saving", lambda v: v.lower().startswith("y")),
    "Solar Model Indicator": ("solar_model_indicator", str),
    "Clearness {0.0 to 1.1}": ("clearness", float),
}

#: Humidity is stated in whichever unit the condition type names.
DDY_HUMIDITY_FIELDS = {
    "Wetbulb at Maximum Dry-Bulb {C}": "wetbulb",
    "Wetbulb or Dewpoint at Maximum Dry-Bulb": "wetbulb_or_dewpoint",
    "Dewpoint at Maximum Dry-Bulb {C}": "dewpoint",
    "Relative Humidity at Maximum Dry-Bulb": "relhum",
    "Humidity Ratio at Maximum Dry-Bulb {kgWater/kgDryAir}": "humidity_ratio",
    "Enthalpy at Maximum Dry-Bulb {J/kg}": "enthalpy",
}

_HUMIDITY_PREFIXES = (
    ("wetbulb", "wetbulb"),
    ("dewpoint", "dewpoint"),
    ("enthalpy", "enthalpy"),
    ("humidityratio", "humidity_ratio"),
    ("relative", "relhum"),
)

_HUMIDITY_FALLBACK_ORDER = [
    "wetbulb", "dewpoint", "relhum", "humidity_ratio", "enthalpy", "wetbulb_or_dewpoint",
]


def extract_design_day_blocks(text: str) -> list[str]:
    """Every ``SizingPeriod:DesignDay`` object in a DDY file, as raw text."""
    return re.findall(r"SizingPeriod:DesignDay,[\s\S]*?;", text)


def parse_design_day_block(block: str) -> dict:
    """Split one design-day object into ``{field comment: value}``."""
    fields = {}
    for line in block.splitlines()[1:]:
        if "!- " not in line:
            continue
        val_part, comment = line.split("!-", 1)
        fields[comment.strip().rstrip(",;")] = val_part.strip().rstrip(",;")
    return fields


def to_number(v, caster):
    """Cast, returning ``None`` for blanks and unparseable values."""
    if v is None or v == "":
        return None
    try:
        return caster(v)
    except ValueError:
        return None


def build_row_from_fields(fields: dict) -> dict:
    """Map one parsed design day onto the flat row shape.

    The humidity value is taken from whichever field matches the stated
    condition type, falling back to the first available in a fixed order.
    """
    row = dict.fromkeys(
        [
            "design_day_name", "day_type", "month", "day_of_month",
            "max_drybulb_c", "daily_range_c",
            "humidity_condition_type", "humidity_value_type", "humidity_value",
            "barometric_pressure_pa", "wind_speed_ms", "wind_direction_deg",
            "rain_flag", "snow_flag", "daylight_saving",
            "solar_model_indicator", "clearness",
        ]
    )
    extra = {}

    for field_name, value in fields.items():
        mapping = DDY_FIELD_MAP.get(field_name)
        if mapping is None:
            extra[field_name] = value
            continue
        col_name, caster = mapping
        row[col_name] = to_number(value, caster) if caster in (int, float) else caster(value)

    candidates = {}
    for field_name, kind in DDY_HUMIDITY_FIELDS.items():
        if field_name not in fields:
            continue
        num = to_number(fields.get(field_name), float)
        if num is not None:
            candidates[kind] = num

    cond = row.get("humidity_condition_type") or fields.get("Humidity Condition Type")
    cond_key = None
    if cond:
        cond_norm = cond.strip().lower()
        cond_key = next((key for pre, key in _HUMIDITY_PREFIXES if cond_norm.startswith(pre)), None)

    selected = None
    if cond_key and cond_key in candidates:
        selected = (cond_key, candidates[cond_key])
    else:
        for key in _HUMIDITY_FALLBACK_ORDER:
            if key in candidates:
                selected = (key, candidates[key])
                break

    if selected is not None:
        row["humidity_value_type"], row["humidity_value"] = selected

    row["extra"] = extra
    return row


def load_ddy_files(ddy_files, keep_raw: bool = False):
    """Parse every DDY into file metadata and one row per design day."""
    files_meta, design_days = [], []

    for file_id, path in enumerate(ddy_files, start=1):
        path = Path(path)
        blocks = extract_design_day_blocks(path.read_text(encoding="utf-8", errors="ignore"))

        files_meta.append(
            {
                "ddy_file_id": file_id,
                "ddy_file_name": path.name,
                "epw_file_name": path.name.replace(".ddy", ".epw"),
            }
        )

        for block in blocks:
            row = build_row_from_fields(parse_design_day_block(block))
            row["file_id"] = file_id
            if keep_raw:
                row["raw_object"] = block
            design_days.append(row)

    return (
        pd.DataFrame(files_meta),
        pd.DataFrame(design_days) if design_days else pd.DataFrame(),
    )


def summarise_ddy_by_city(
    ddy_files_df: pd.DataFrame, epw_files_df: pd.DataFrame, design_days_df: pd.DataFrame
) -> pd.DataFrame:
    """Reduce design days to summer and winter aggregates per city.

    ``summer_humidity_value_mean`` averages across humidity condition types; see
    the module docstring.
    """
    if ddy_files_df.empty:
        return pd.DataFrame()

    epw_meta = epw_files_df.rename(
        columns={"file_id": "epw_file_id", "file_name": "epw_file_name"}
    )
    ddy_epw_df = ddy_files_df.merge(
        epw_meta[
            [
                "epw_file_id", "epw_file_name", "country", "state_province", "source",
                "wmo", "latitude", "longitude", "time_zone", "elevation_m",
            ]
        ],
        on="epw_file_name",
        how="left",
    )
    dd_merged = ddy_epw_df.merge(
        design_days_df, left_on="ddy_file_id", right_on="file_id", how="left"
    )

    group_cols = [
        "ddy_file_id", "ddy_file_name", "epw_file_id", "epw_file_name",
        "country", "state_province", "source", "wmo",
        "latitude", "longitude", "time_zone", "elevation_m",
    ]

    def summarize_group(g):
        summer = g[g["day_type"] == "SummerDesignDay"]
        winter = g[g["day_type"] == "WinterDesignDay"]
        return pd.Series(
            {
                "n_summer_design_days": len(summer),
                "n_winter_design_days": len(winter),
                "summer_max_drybulb_c": summer["max_drybulb_c"].max(),
                "summer_min_drybulb_c": summer["max_drybulb_c"].min(),
                "summer_mean_max_drybulb_c": summer["max_drybulb_c"].mean(),
                "winter_min_drybulb_c": winter["max_drybulb_c"].min(),
                "winter_max_drybulb_c": winter["max_drybulb_c"].max(),
                "winter_mean_max_drybulb_c": winter["max_drybulb_c"].mean(),
                "summer_daily_range_mean_c": summer["daily_range_c"].mean(),
                "winter_daily_range_mean_c": winter["daily_range_c"].mean(),
                "summer_humidity_value_mean": summer["humidity_value"].mean(),
                "winter_humidity_value_mean": winter["humidity_value"].mean(),
                "summer_wind_speed_mean_ms": summer["wind_speed_ms"].mean(),
                "winter_wind_speed_mean_ms": winter["wind_speed_ms"].mean(),
                "summer_wind_direction_mean_deg": summer["wind_direction_deg"].mean(),
                "winter_wind_direction_mean_deg": winter["wind_direction_deg"].mean(),
            }
        )

    return (
        dd_merged.groupby(group_cols, dropna=False)[dd_merged.columns]
        .apply(summarize_group)
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_weather_city(
    epw_city_df: pd.DataFrame, ddy_city_df: pd.DataFrame
) -> pd.DataFrame:
    """Join the EPW and DDY city summaries into one row per ``city_key``."""
    if epw_city_df.empty and ddy_city_df.empty:
        return pd.DataFrame()

    epw_df = epw_city_df.copy()
    ddy_df = ddy_city_df.copy()
    if not ddy_df.empty:
        ddy_df = ddy_df.rename(columns={"epw_file_name": "epw_file_name_ddy"})

    merged = epw_df.merge(
        ddy_df,
        left_on="epw_file_name",
        right_on="epw_file_name_ddy",
        how="outer",
        suffixes=("", "_ddy"),
    )

    city_key = merged["epw_file_name"].str.replace(".epw", "", regex=False)
    if "ddy_file_name" in merged:
        city_key = city_key.combine_first(
            merged["ddy_file_name"].str.replace(".ddy", "", regex=False)
        )
    if "epw_file_name_ddy" in merged:
        city_key = city_key.combine_first(
            merged["epw_file_name_ddy"].str.replace(".epw", "", regex=False)
        )

    def prefer(col):
        alt = f"{col}_ddy"
        return merged[col].combine_first(merged[alt]) if alt in merged.columns else merged[col]

    exclude_epw = set(EPW_BASE_COLS)
    exclude_ddy = {
        "ddy_file_id", "ddy_file_name", "epw_file_id", "epw_file_name_ddy",
        "country", "state_province", "source", "wmo",
        "latitude", "longitude", "time_zone", "elevation_m",
        "latitude_ddy", "longitude_ddy", "time_zone_ddy", "elevation_m_ddy",
    }
    epw_metric_cols = [c for c in epw_df.columns if c not in exclude_epw]
    ddy_metric_cols = [c for c in ddy_df.columns if c not in exclude_ddy]

    out = merged.copy()
    out["city_key"] = city_key.str.lower()
    out["latitude"] = prefer("latitude")
    out["longitude"] = prefer("longitude")
    out["elevation_m"] = prefer("elevation_m")
    out["time_zone"] = prefer("time_zone").map(
        lambda v: None if pd.isna(v) else f"UTC{float(v):+g}"
    )

    keep_cols = [
        "city_key", "epw_file_name", "ddy_file_name", "country", "state_province",
        "source", "wmo", "time_zone", "latitude", "longitude", "elevation_m",
    ] + epw_metric_cols + ddy_metric_cols
    out = out[[c for c in keep_cols if c in out.columns]]

    if out["city_key"].duplicated().any():
        dups = out.loc[out["city_key"].duplicated(), "city_key"].tolist()[:10]
        raise RuntimeError(f"Duplicate city_key rows in weather_city: {dups}")

    return out


def city_key_for_model(model_id: str) -> str:
    """``'A_Coruna_0001'`` -> ``'a_coruna'``: strip the trailing number, lower-case."""
    return re.sub(r"_[0-9]+$", "", model_id).lower()


@dataclass
class WeatherWriteResult:
    """What was written to ``ep.weather_city``."""

    rows: int = 0
    cities: int = 0
    columns: list = field(repr=False, default_factory=list)
    missing_models: list = field(repr=False, default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "weather_city_rows": self.rows,
                    "distinct_cities": self.cities,
                    "weather_columns": len(self.columns),
                    "models_without_weather": len(self.missing_models),
                }
            ]
        )


def write_weather_city(
    engine, weather_city_df: pd.DataFrame, schema: str, strict: bool = True
) -> WeatherWriteResult:
    """Expand the per-city table to one row per model and write ``ep.weather_city``.

    Every model gets a row, so downstream transforms can join on ``model_id``
    alone. With ``strict=True`` a model that resolves to no city raises rather
    than silently producing a row of NULLs.
    """
    if weather_city_df.empty:
        raise RuntimeError("weather_city_df is empty; refusing to write the table.")

    model_ids = pd.read_sql(f'SELECT model_id FROM {schema}.ep_model', engine)
    if model_ids.empty:
        raise RuntimeError(f"{schema}.ep_model is empty; ingest the models first.")

    model_ids["city_key"] = model_ids["model_id"].map(city_key_for_model)
    per_model = model_ids.merge(weather_city_df, on="city_key", how="left")
    per_model.insert(1, "instance_name", per_model["city_key"])

    weather_cols = weather_city_df.columns.tolist()
    value_cols = [c for c in weather_cols if c != "city_key"]
    per_model = per_model[["model_id", "instance_name"] + weather_cols]

    missing_mask = per_model[value_cols].isna().all(axis=1)
    missing = per_model.loc[missing_mask, "model_id"].tolist()
    if missing and strict:
        raise AssertionError(
            f"{len(missing)} model(s) have no weather row. "
            f"Sample: {', '.join(missing[:10])}"
        )

    if per_model["model_id"].duplicated().any():
        dups = per_model.loc[per_model["model_id"].duplicated(), "model_id"].tolist()[:10]
        raise AssertionError(f"Duplicate model_id rows in the weather_city payload: {dups}")

    # Drop an existing weather_city, view or table, together with its dependents
    with engine.begin() as conn:
        relkind = conn.execute(
            sa_text(
                """
            SELECT c.relkind FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname = 'weather_city'
        """
            ),
            {"schema": schema},
        ).scalar()
        if relkind == "v":
            conn.execute(sa_text(f"DROP VIEW IF EXISTS {schema}.weather_city CASCADE"))
        elif relkind in {"r", "p"}:
            conn.execute(sa_text(f"DROP TABLE IF EXISTS {schema}.weather_city CASCADE"))

    per_model.to_sql("weather_city", engine, schema=schema, if_exists="replace", index=False)

    ensure_standard_relation_indexes(engine, schema, "weather_city", verbose=False)
    ensure_btree_index(
        engine,
        schema,
        "weather_city",
        ["city_key"],
        index_name="idx_weather_city_city_key",
        verbose=False,
    )

    return WeatherWriteResult(
        rows=len(per_model),
        cities=int(per_model["city_key"].nunique()),
        columns=weather_cols,
        missing_models=missing,
    )


def build_and_write(
    engine, epw_files, ddy_files, schema: str, keep_hourly: bool = False, strict: bool = True
):
    """Run the whole weather stage: parse, summarise, assemble, write.

    Returns ``(weather_city_df, WeatherWriteResult)``.
    """
    epw_files_df, epw_monthly_df, _hourly = load_epw_files(epw_files, keep_hourly=keep_hourly)
    ddy_files_df, design_days_df = load_ddy_files(ddy_files)

    epw_city = summarise_epw_by_city(epw_monthly_df)
    ddy_city = summarise_ddy_by_city(ddy_files_df, epw_files_df, design_days_df)

    weather_city_df = build_weather_city(epw_city, ddy_city)
    return weather_city_df, write_weather_city(engine, weather_city_df, schema, strict=strict)
