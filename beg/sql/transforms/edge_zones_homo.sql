-- ============================================================================
-- edge_zones_homo.sql (Zone adjacency for the homogeneous zone configuration)
--   • The same zone-to-zone adjacency as edge_zones, with no edge features.
--   • Used by C1, the homogeneous zone configuration: one node type and one edge
--     type, and nothing on an edge for a model to condition on. C3/C4/C5 read
--     edge_zones, which carries `shared_boundary_area`.
--   • Mirrors C6, whose edge relation likewise carries no features.
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones_homo CASCADE;

CREATE VIEW ep.edge_zones_homo AS
SELECT
  model_id,
  zone_1_name,
  zone_2_name
FROM ep.edge_zones;
