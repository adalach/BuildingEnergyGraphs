-- ============================================================================
-- building_weather.sql
--   • Creates view ep.building_weather: ep.building plus the ep.weather_city columns
--   • Includes all building properties first, then appends weather columns
--   • Match key: model_id (ep.weather_city is stored per model_id)
-- ============================================================================

DROP VIEW IF EXISTS ep.building_weather CASCADE;

CREATE VIEW ep.building_weather AS
SELECT
  -- --------------------------------------------------------------------------
  -- All building properties first
  -- --------------------------------------------------------------------------
  b.*,

  -- --------------------------------------------------------------------------
  -- Then the weather columns
  -- --------------------------------------------------------------------------
  wc.city_key,
  wc.latitude,
  wc.longitude,
  wc.elevation_m,
  -- All remaining weather_city columns
  wc.n_summer_design_days,
  wc.n_winter_design_days,
  wc.summer_max_drybulb_c,
  wc.summer_min_drybulb_c,
  wc.summer_mean_max_drybulb_c,
  wc.winter_min_drybulb_c,
  wc.winter_max_drybulb_c,
  wc.winter_mean_max_drybulb_c,
  wc.summer_daily_range_mean_c,
  wc.winter_daily_range_mean_c,
  wc.summer_humidity_value_mean,
  wc.winter_humidity_value_mean,
  wc.summer_wind_speed_mean_ms,
  wc.winter_wind_speed_mean_ms,
  wc.summer_wind_direction_mean_deg,
  wc.winter_wind_direction_mean_deg,
  -- EPW-wide climate descriptors (pivoted monthly metrics)
  wc.t_mean_c_01,  wc.t_mean_c_02,  wc.t_mean_c_03,  wc.t_mean_c_04,
  wc.t_mean_c_05,  wc.t_mean_c_06,  wc.t_mean_c_07,  wc.t_mean_c_08,
  wc.t_mean_c_09,  wc.t_mean_c_10,  wc.t_mean_c_11,  wc.t_mean_c_12,
  wc.t_min_c_01,   wc.t_min_c_02,   wc.t_min_c_03,   wc.t_min_c_04,
  wc.t_min_c_05,   wc.t_min_c_06,   wc.t_min_c_07,   wc.t_min_c_08,
  wc.t_min_c_09,   wc.t_min_c_10,   wc.t_min_c_11,   wc.t_min_c_12,
  wc.t_max_c_01,   wc.t_max_c_02,   wc.t_max_c_03,   wc.t_max_c_04,
  wc.t_max_c_05,   wc.t_max_c_06,   wc.t_max_c_07,   wc.t_max_c_08,
  wc.t_max_c_09,   wc.t_max_c_10,   wc.t_max_c_11,   wc.t_max_c_12,
  wc.t_day_mean_c_01,  wc.t_day_mean_c_02,  wc.t_day_mean_c_03,  wc.t_day_mean_c_04,
  wc.t_day_mean_c_05,  wc.t_day_mean_c_06,  wc.t_day_mean_c_07,  wc.t_day_mean_c_08,
  wc.t_day_mean_c_09,  wc.t_day_mean_c_10,  wc.t_day_mean_c_11,  wc.t_day_mean_c_12,
  wc.t_night_mean_c_01, wc.t_night_mean_c_02, wc.t_night_mean_c_03, wc.t_night_mean_c_04,
  wc.t_night_mean_c_05, wc.t_night_mean_c_06, wc.t_night_mean_c_07, wc.t_night_mean_c_08,
  wc.t_night_mean_c_09, wc.t_night_mean_c_10, wc.t_night_mean_c_11, wc.t_night_mean_c_12,
  wc.rh_mean_pct_01,  wc.rh_mean_pct_02,  wc.rh_mean_pct_03,  wc.rh_mean_pct_04,
  wc.rh_mean_pct_05,  wc.rh_mean_pct_06,  wc.rh_mean_pct_07,  wc.rh_mean_pct_08,
  wc.rh_mean_pct_09,  wc.rh_mean_pct_10,  wc.rh_mean_pct_11,  wc.rh_mean_pct_12,
  wc.rh80_hours_01,   wc.rh80_hours_02,   wc.rh80_hours_03,   wc.rh80_hours_04,
  wc.rh80_hours_05,   wc.rh80_hours_06,   wc.rh80_hours_07,   wc.rh80_hours_08,
  wc.rh80_hours_09,   wc.rh80_hours_10,   wc.rh80_hours_11,   wc.rh80_hours_12,
  wc.wind_speed_mean_ms_01, wc.wind_speed_mean_ms_02, wc.wind_speed_mean_ms_03, wc.wind_speed_mean_ms_04,
  wc.wind_speed_mean_ms_05, wc.wind_speed_mean_ms_06, wc.wind_speed_mean_ms_07, wc.wind_speed_mean_ms_08,
  wc.wind_speed_mean_ms_09, wc.wind_speed_mean_ms_10, wc.wind_speed_mean_ms_11, wc.wind_speed_mean_ms_12,
  wc.strong_wind_hours_01,  wc.strong_wind_hours_02,  wc.strong_wind_hours_03,  wc.strong_wind_hours_04,
  wc.strong_wind_hours_05,  wc.strong_wind_hours_06,  wc.strong_wind_hours_07,  wc.strong_wind_hours_08,
  wc.strong_wind_hours_09,  wc.strong_wind_hours_10,  wc.strong_wind_hours_11,  wc.strong_wind_hours_12,
  wc.ghi_kwh_m2_01, wc.ghi_kwh_m2_02, wc.ghi_kwh_m2_03, wc.ghi_kwh_m2_04,
  wc.ghi_kwh_m2_05, wc.ghi_kwh_m2_06, wc.ghi_kwh_m2_07, wc.ghi_kwh_m2_08,
  wc.ghi_kwh_m2_09, wc.ghi_kwh_m2_10, wc.ghi_kwh_m2_11, wc.ghi_kwh_m2_12,
  wc.dni_kwh_m2_01, wc.dni_kwh_m2_02, wc.dni_kwh_m2_03, wc.dni_kwh_m2_04,
  wc.dni_kwh_m2_05, wc.dni_kwh_m2_06, wc.dni_kwh_m2_07, wc.dni_kwh_m2_08,
  wc.dni_kwh_m2_09, wc.dni_kwh_m2_10, wc.dni_kwh_m2_11, wc.dni_kwh_m2_12,
  wc.dhi_kwh_m2_01, wc.dhi_kwh_m2_02, wc.dhi_kwh_m2_03, wc.dhi_kwh_m2_04,
  wc.dhi_kwh_m2_05, wc.dhi_kwh_m2_06, wc.dhi_kwh_m2_07, wc.dhi_kwh_m2_08,
  wc.dhi_kwh_m2_09, wc.dhi_kwh_m2_10, wc.dhi_kwh_m2_11, wc.dhi_kwh_m2_12,
  wc.daylight_hours_01, wc.daylight_hours_02, wc.daylight_hours_03, wc.daylight_hours_04,
  wc.daylight_hours_05, wc.daylight_hours_06, wc.daylight_hours_07, wc.daylight_hours_08,
  wc.daylight_hours_09, wc.daylight_hours_10, wc.daylight_hours_11, wc.daylight_hours_12,
  wc.precip_mm_01, wc.precip_mm_02, wc.precip_mm_03, wc.precip_mm_04,
  wc.precip_mm_05, wc.precip_mm_06, wc.precip_mm_07, wc.precip_mm_08,
  wc.precip_mm_09, wc.precip_mm_10, wc.precip_mm_11, wc.precip_mm_12,
  wc.precip_hours_01, wc.precip_hours_02, wc.precip_hours_03, wc.precip_hours_04,
  wc.precip_hours_05, wc.precip_hours_06, wc.precip_hours_07, wc.precip_hours_08,
  wc.precip_hours_09, wc.precip_hours_10, wc.precip_hours_11, wc.precip_hours_12,
  wc.hdd18_01, wc.hdd18_02, wc.hdd18_03, wc.hdd18_04,
  wc.hdd18_05, wc.hdd18_06, wc.hdd18_07, wc.hdd18_08,
  wc.hdd18_09, wc.hdd18_10, wc.hdd18_11, wc.hdd18_12,
  wc.cdd26_01, wc.cdd26_02, wc.cdd26_03, wc.cdd26_04,
  wc.cdd26_05, wc.cdd26_06, wc.cdd26_07, wc.cdd26_08,
  wc.cdd26_09, wc.cdd26_10, wc.cdd26_11, wc.cdd26_12

FROM ep.building AS b
LEFT JOIN ep.weather_city AS wc
    ON wc.model_id = b.model_id;
