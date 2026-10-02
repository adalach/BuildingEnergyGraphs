-- ============================================================================
-- edge_comp_zone_building_weather.sql
--   • Creates explicit edges between comp_zone nodes and building_weather nodes
--   • Links zones to their parent building via model_id
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_zone_building_weather CASCADE;

CREATE VIEW ep.edge_comp_zone_building_weather AS
SELECT
  cz.model_id       AS model_id,
  cz.instance_name  AS zone_instance_name,
  bw.instance_name  AS building_weather_instance_name
FROM ep.comp_zone cz
JOIN ep.building_weather bw
  ON cz.model_id = bw.model_id;
