-- ============================================================================
-- edge_comp_aperture_element.sql
--   • Creates edges between comp_aperture nodes and comp_element nodes
--   • Uses building_surface_name from aperture to link to instance_name in element
--   • Relationship: comp_aperture belongs to comp_element (parent surface)
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_aperture_element CASCADE;

CREATE VIEW ep.edge_comp_aperture_element AS
SELECT
  a.model_id,
  a.instance_name AS aperture_name,
  e.instance_name AS element_name,
  a.surface_type AS aperture_type
FROM ep.comp_aperture a
JOIN ep.comp_element e
  ON a.model_id = e.model_id
  AND a.building_surface_name = e.instance_name
WHERE a.building_surface_name IS NOT NULL
ORDER BY a.model_id, a.instance_name, e.instance_name;

