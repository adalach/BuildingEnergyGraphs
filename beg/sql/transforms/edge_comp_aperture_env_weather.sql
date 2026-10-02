-- ============================================================================
-- edge_comp_aperture_env_weather.sql
--   • Creates edges between external window apertures and weather_city nodes
--   • Apertures included:
--       - windows whose parent surface is an external wall
--       - windows whose parent surface is an external roof/ceiling
--   • Weather node: ep.weather_city (model_id → city_key)
--   • Output view:
--       - ep.edge_comp_aperture_env_weather (comp_aperture(window) → weather_city)
--   • Uses double precision for areas and float-safe rounding
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_aperture_env_weather CASCADE;

CREATE VIEW ep.edge_comp_aperture_env_weather AS
SELECT
  a.model_id,
  a.instance_name AS aperture_name,
  w.city_key,
  CASE WHEN p.is_wall = 1 AND p.is_outdoors = 1 THEN 1 ELSE 0 END::int AS is_parent_ext_wall,
  CASE WHEN (p.is_roof = 1 OR p.is_ceiling = 1) AND p.is_outdoors = 1 THEN 1 ELSE 0 END::int AS is_parent_ext_roof,
  (ROUND(COALESCE(a.area, 0)::double precision * 1e4) / 1e4) AS aperture_area
FROM ep.comp_aperture a
JOIN ep.comp_element p
  ON p.model_id = a.model_id
 AND p.instance_name = a.building_surface_name
JOIN ep.weather_city w
  ON w.model_id = a.model_id
WHERE a.is_window = 1
  AND (
    (p.is_wall = 1 AND p.is_outdoors = 1)
    OR ((p.is_roof = 1 OR p.is_ceiling = 1) AND p.is_outdoors = 1)
  );
