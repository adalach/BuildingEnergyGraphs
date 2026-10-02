-- ============================================================================
-- edge_comp_base_unified.sql (Unified Edge View with Multiple Relationship Types)
--   • Combines three types of relationships:
--     1. same_zone: Components in the same thermal zone
--     2. adjacency: Adjacent building surfaces across zones
--     3. aperture_to_element: Apertures (windows/doors) connected to their parent surfaces
--   • C6 and C7 use it as their one edge type; no edge feature tells the three relations
--     apart.
--   • `relationship_type` is text, kept for inspection: the graph builders use only
--     numeric columns as features.
--   • All edges are bidirectional (mirroring is handled by the graph builder / pipeline rules)
--   • Performance-optimized: avoids large self-join on comp_base_element, narrows EXISTS checks
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_comp_base_unified CASCADE;

CREATE VIEW ep.edge_comp_base_unified AS

/* 1) Same-Zone Edges: Components in the same thermal zone (fast pair gen) */
WITH same_zone_edges AS (
  WITH zone_elems AS (
    SELECT
      e.model_id,
      e.zone_name,
      array_agg(e.instance_name ORDER BY e.instance_name) AS elems
    FROM ep.comp_element e
    WHERE e.zone_name IS NOT NULL
      AND (
        e.is_wall = 1 OR e.is_floor = 1 OR e.is_roof = 1 OR e.is_ceiling = 1
      )
    GROUP BY e.model_id, e.zone_name
  )
  SELECT
    z.model_id,
    z.elems[i] AS element_1_name,
    z.elems[j] AS element_2_name,
    'same_zone' AS relationship_type
  FROM zone_elems z
  JOIN LATERAL generate_subscripts(z.elems, 1) AS i ON TRUE
  JOIN LATERAL generate_subscripts(z.elems, 1) AS j ON j > i
),

/* 2) Adjacency Edges: Adjacent surfaces via outside_boundary_condition ---- */
adjacency_edges AS (
  SELECT
    s1.model_id,
    s1.instance_name AS element_1_name,
    s2.instance_name AS element_2_name,
    'adjacency' AS relationship_type
  FROM ep.ep_buildingsurface_detailed s1
  JOIN ep.ep_buildingsurface_detailed s2
    ON s1.model_id = s2.model_id
   AND s1.outside_boundary_condition_object = s2.instance_name
   AND s1.outside_boundary_condition_surface = 1  -- Has Surface boundary condition
   AND s1.outside_boundary_condition_object IS NOT NULL
   AND TRIM(s1.outside_boundary_condition_object) <> ''
   AND s1.instance_name < s2.instance_name  -- Avoid duplicates
  -- Only include surfaces that are present among component elements
  WHERE EXISTS (
    SELECT 1 FROM ep.comp_element e
     WHERE e.model_id = s1.model_id AND e.instance_name = s1.instance_name
  )
    AND EXISTS (
    SELECT 1 FROM ep.comp_element e
     WHERE e.model_id = s2.model_id AND e.instance_name = s2.instance_name
  )
),

/* 3) Aperture-to-Element Edges: Windows/doors to their parent surfaces ---- */
aperture_to_element_edges AS (
  SELECT
    a.model_id,
    a.instance_name AS element_1_name,
    a.building_surface_name AS element_2_name,
    'aperture_to_element' AS relationship_type
  FROM ep.comp_aperture a
  WHERE a.building_surface_name IS NOT NULL
    AND TRIM(a.building_surface_name) <> ''
    -- Parent surface must be a component element
    AND EXISTS (
      SELECT 1 FROM ep.comp_element e
       WHERE e.model_id = a.model_id AND e.instance_name = a.building_surface_name
    )
    -- Aperture must exist among apertures
    AND EXISTS (
      SELECT 1 FROM ep.comp_aperture ax
       WHERE ax.model_id = a.model_id AND ax.instance_name = a.instance_name
    )
)

/* 4) Union all edge types -------------------------------------------------- */
SELECT * FROM same_zone_edges
UNION ALL
SELECT * FROM adjacency_edges
UNION ALL
SELECT * FROM aperture_to_element_edges;
