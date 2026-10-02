-- ============================================================================
-- edge_comp_zone_building.sql
--   • Creates explicit edges between comp_zone nodes and building nodes
--   • Links zones to their parent building via model_id
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_zone_building CASCADE;

CREATE VIEW ep.edge_comp_zone_building AS
SELECT
  cz.model_id,
  cz.instance_name  AS zone_instance_name,
  b.instance_name   AS building_instance_name
FROM ep.comp_zone cz
JOIN ep.building b
  ON cz.model_id = b.model_id;
