-- ============================================================================
-- edge_comp_element_zone.sql
--   • Creates edges between comp_element nodes and comp_zone nodes
--   • Uses zone_name from comp_element to link to instance_name in comp_zone
--   • Relationship: comp_element belongs to zone
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_element_zone CASCADE;

CREATE VIEW ep.edge_comp_element_zone AS
SELECT
  c.model_id,
  c.instance_name AS comp_element_name,
  z.instance_name AS zone_name,
  c.construction_name AS element_construction
FROM ep.comp_element c
JOIN ep.comp_zone z
  ON c.model_id = z.model_id
  AND c.zone_name = z.instance_name
WHERE c.zone_name IS NOT NULL
ORDER BY c.model_id, c.instance_name, z.instance_name;

