-- ============================================================================
-- edge_comp_element_adjacency.sql
--   • Creates edges between adjacent comp_element nodes
--   • Uses outside_boundary_condition_object from ep_buildingsurface_detailed
--   • Links surfaces where outside_boundary_condition_object references another surface
--   • Handles empty strings and NULL values
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_element_adjacency CASCADE;

CREATE VIEW ep.edge_comp_element_adjacency AS
WITH
/* 1) Get surfaces with non-empty outside_boundary_condition_object -------- */
surfaces_with_adjacency AS (
  SELECT
    b.id,
    b.model_id,
    b.instance_name,
    b.zone_name,
    b.construction_name,
    b.outside_boundary_condition_object,
    b.outside_boundary_condition_surface
  FROM ep.ep_buildingsurface_detailed b
  WHERE b.outside_boundary_condition_surface = 1  -- Has Surface boundary condition
    AND b.outside_boundary_condition_object IS NOT NULL
    AND TRIM(b.outside_boundary_condition_object) != ''
),

/* 2) Join with comp_element to get element IDs ---------------------------- */
elements_with_adjacency AS (
  SELECT
    e.id AS element_id,
    e.model_id,
    e.instance_name AS element_name,
    e.zone_name AS element_zone,
    e.construction_name AS element_construction,
    s.outside_boundary_condition_object AS adjacent_surface_name
  FROM ep.comp_element e
  JOIN surfaces_with_adjacency s
    ON e.model_id = s.model_id
    AND e.instance_name = s.instance_name
),

/* 3) Create adjacency pairs ----------------------------------------------- */
adjacency_pairs AS (
  SELECT
    e1.model_id,
    e1.element_name AS element_1_name,
    e2.element_name AS element_2_name,
    e1.element_zone AS element_1_zone,
    e2.element_zone AS element_2_zone,
    e1.element_construction AS element_1_construction,
    e2.element_construction AS element_2_construction
  FROM elements_with_adjacency e1
  JOIN elements_with_adjacency e2
    ON e1.model_id = e2.model_id
    AND e1.adjacent_surface_name = e2.element_name
  WHERE e1.element_name < e2.element_name  -- Avoid duplicates (keep only one direction)
)

/* 4) Final output ---------------------------------------------------------- */
SELECT
  model_id,
  element_1_name,
  element_2_name,
  CASE WHEN element_1_zone != element_2_zone THEN 1 ELSE 0 END::int AS is_cross_zone
FROM adjacency_pairs
ORDER BY model_id, element_1_name, element_2_name;
