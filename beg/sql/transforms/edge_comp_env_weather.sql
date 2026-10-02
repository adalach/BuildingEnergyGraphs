-- ============================================================================
-- edge_comp_env_weather.sql
--   • Creates edges between envelope components and weather_city nodes
--   • Envelope components include:
--       - external walls:   is_wall=1 AND is_outdoors=1
--       - roofs/ceilings:   (is_roof=1 OR is_ceiling=1) AND is_outdoors=1
--       - ground floors:    is_floor=1 AND is_ground=1
--   • Weather node: ep.weather_city (model_id → city_key)
--   • Output view:
--       - ep.edge_comp_env_weather (comp_element → weather_city)
--   • Uses double precision for areas and float-safe rounding
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_env_weather CASCADE;

CREATE VIEW ep.edge_comp_env_weather AS
SELECT
  e.model_id,
  e.instance_name AS element_name,
  w.city_key,
  CASE WHEN e.is_wall = 1 AND e.is_outdoors = 1 THEN 1 ELSE 0 END::int AS is_ext_wall,
  CASE WHEN (e.is_roof = 1 OR e.is_ceiling = 1) AND e.is_outdoors = 1 THEN 1 ELSE 0 END::int AS is_ext_roof,
  CASE WHEN e.is_floor = 1 AND e.is_ground = 1 THEN 1 ELSE 0 END::int AS is_ground_floor,
  (ROUND(COALESCE(e.area, 0)::double precision * 1e4) / 1e4) AS surface_area
FROM ep.comp_element e
JOIN ep.weather_city w
  ON w.model_id = e.model_id
WHERE
  (e.is_wall = 1 AND e.is_outdoors = 1)
  OR ((e.is_roof = 1 OR e.is_ceiling = 1) AND e.is_outdoors = 1)
  OR (e.is_floor = 1 AND e.is_ground = 1);
