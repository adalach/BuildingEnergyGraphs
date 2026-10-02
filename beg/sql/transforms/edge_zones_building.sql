-- ============================================================================
-- edge_zones_building.sql
--   • Creates explicit edges between zones_zone nodes and building nodes
--   • Links zones to their parent building via model_id
--   • Column names are aligned with configs:
--       - zone_name (from zones_zone.instance_name)
--       - building_instance_name (from building.instance_name)
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones_building CASCADE;

CREATE VIEW ep.edge_zones_building AS
SELECT
  z.model_id,
  z.instance_name      AS zone_name,
  b.instance_name      AS building_instance_name
FROM ep.zones_zone z
JOIN ep.building b
  ON b.model_id = z.model_id;
