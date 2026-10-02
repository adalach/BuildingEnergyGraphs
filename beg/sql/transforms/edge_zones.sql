-- ============================================================================
-- edge_zones.sql
--   • Creates edges between adjacent zones (zones that share an adjacent surface)
--   • Optimized to strictly join within the same model_id
--   • Filters early to reduce join cardinality
--   • Uses LATERAL join to avoid large self-join explosions
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones CASCADE;

CREATE VIEW ep.edge_zones AS
WITH
/* 1) Only surfaces that actually participate in zone adjacency ------------- */
surfaces AS (
  SELECT
    b.model_id,
    b.instance_name AS surface_name,
    b.zone_name,
    b.outside_boundary_condition_object,
    COALESCE(b.area, 0)::double precision AS area
  FROM ep.buildingsurface_features b
  WHERE b.zone_name IS NOT NULL
    AND b.outside_boundary_condition = 'Surface'
),

/* 2) Match each surface only to its referenced adjacent surface
      within the SAME model (forced by LATERAL) ----------------------------- */
surface_pairs AS (
  SELECT
    s1.model_id,
    LEAST(s1.zone_name, s2.zone_name) AS zone_1_name,
    GREATEST(s1.zone_name, s2.zone_name) AS zone_2_name,
    LEAST(s1.surface_name, s2.surface_name) AS surface_1_name,
    GREATEST(s1.surface_name, s2.surface_name) AS surface_2_name,
    CASE
      WHEN s1.area > 0 AND s2.area > 0 THEN (s1.area + s2.area) / 2.0
      WHEN s1.area > 0 THEN s1.area
      WHEN s2.area > 0 THEN s2.area
      ELSE 0.0
    END AS interface_area
  FROM surfaces s1
  JOIN LATERAL (
    SELECT
      s2.zone_name,
      s2.surface_name,
      s2.area
    FROM surfaces s2
    WHERE s2.model_id = s1.model_id
      AND s2.surface_name = s1.outside_boundary_condition_object
  ) s2 ON true
  WHERE s1.zone_name <> s2.zone_name
),

/* 3) Deduplicate undirected surface pairs ---------------------------------- */
dedup AS (
  SELECT DISTINCT
    model_id,
    zone_1_name,
    zone_2_name,
    surface_1_name,
    surface_2_name,
    interface_area
  FROM surface_pairs
)

/* 4) Aggregate per zone–zone edge ------------------------------------------ */
SELECT
  model_id,
  zone_1_name,
  zone_2_name,
  ROUND(SUM(interface_area) * 1e4) / 1e4 AS shared_boundary_area
FROM dedup
GROUP BY model_id, zone_1_name, zone_2_name
ORDER BY model_id, zone_1_name, zone_2_name;

