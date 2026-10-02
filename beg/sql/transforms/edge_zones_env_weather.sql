-- ============================================================================
-- edge_zones_env_weather.sql
--   • Creates edges between external zones (zones_zone) and weather_city nodes
--   • External zones are defined by ep.edge_zones_model_weather (exposure filter)
--   • Joins to ep.weather_city by model_id to attach city_key
--   • Output columns:
--       - model_id
--       - zone_name
--       - city_key
--       - exposure flags + areas (copied through for optional edge attributes)
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones_env_weather CASCADE;

CREATE VIEW ep.edge_zones_env_weather AS
SELECT
  z.model_id,
  z.zone_name,
  w.city_key,
  z.has_ext_wall,
  z.has_ext_roof,
  z.has_ground_contact,
  z.n_external_surfaces,
  z.external_surface_area
FROM ep.edge_zones_model_weather z
JOIN ep.weather_city w
  ON w.model_id = z.model_id
ORDER BY z.model_id, z.zone_name;

