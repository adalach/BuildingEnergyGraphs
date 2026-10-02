-- ============================================================================
-- edge_zones_model_weather_all.sql
--   • Creates edges between zones_zone nodes and building_weather nodes
--   • Links ALL zones to the model-level weather node (no exterior filter)
--   • Still computes exterior exposure flags for convenience:
--     - external wall:   surface_type=Wall   AND outside_boundary_condition_code=1 (Outdoors)
--     - roof/ceiling:    surface_type=Roof/Ceiling AND outside_boundary_condition_code=1 (Outdoors)
--     - ground-contact:  surface_type=Floor  AND outside_boundary_condition_code=3 (Ground)
--   • Uses ep.buildingsurface_features for classification and area
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones_model_weather_all CASCADE;

CREATE VIEW ep.edge_zones_model_weather_all AS
WITH
surfaces AS (
  SELECT
    b.model_id,
    COALESCE(NULLIF(btrim(b.zone_name), ''), NULLIF(btrim(sp.zone_name), '')) AS zone_name,
    LOWER(b.surface_type) AS surface_type,
    b.outside_boundary_condition,
    b.outside_boundary_condition_code,
    COALESCE(b.area, 0)::double precision AS area

  FROM ep.buildingsurface_features b
  LEFT JOIN ep.ep_space sp
    ON sp.model_id = b.model_id
   AND sp.instance_name = NULLIF(btrim(b.space_name), '')
  WHERE COALESCE(NULLIF(btrim(b.zone_name), ''), NULLIF(btrim(sp.zone_name), '')) IS NOT NULL
),
zone_external AS (
  SELECT
    model_id,
    zone_name,
    MAX(CASE
          WHEN surface_type = 'wall'
           AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors')
          THEN 1 ELSE 0
        END)::int AS has_ext_wall,
    MAX(CASE
          WHEN surface_type IN ('roof', 'ceiling', 'roofceiling')
           AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors')
          THEN 1 ELSE 0
        END)::int AS has_ext_roof,
    MAX(CASE
          WHEN surface_type = 'floor'
           AND (
             outside_boundary_condition_code = 3
             OR LOWER(outside_boundary_condition) IN (
               'ground',
               'groundfcfactormethod',
               'groundslabpreprocessoraverage',
               'groundslabpreprocessorcore',
               'groundslabpreprocessorperimeter',
               'groundbasementpreprocessoraveragefloor',
               'groundbasementpreprocessoraveragewall',
               'groundbasementpreprocessorlowerwall',
               'groundbasementpreprocessorupperwall',
               'foundation'
             )
           )
          THEN 1 ELSE 0
        END)::int AS has_ground_contact,
    SUM(
      CASE
        WHEN surface_type = 'wall' AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors') THEN 1
        WHEN surface_type IN ('roof', 'ceiling', 'roofceiling') AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors') THEN 1
        WHEN surface_type = 'floor' AND (
          outside_boundary_condition_code = 3
          OR LOWER(outside_boundary_condition) IN (
            'ground',
            'groundfcfactormethod',
            'groundslabpreprocessoraverage',
            'groundslabpreprocessorcore',
            'groundslabpreprocessorperimeter',
            'groundbasementpreprocessoraveragefloor',
            'groundbasementpreprocessoraveragewall',
            'groundbasementpreprocessorlowerwall',
            'groundbasementpreprocessorupperwall',
            'foundation'
          )
        ) THEN 1
        ELSE 0
      END
    )::int AS n_external_surfaces,
    (ROUND(SUM(
      CASE
        WHEN surface_type = 'wall' AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors') THEN area
        WHEN surface_type IN ('roof', 'ceiling', 'roofceiling') AND (outside_boundary_condition_code = 1 OR LOWER(outside_boundary_condition) = 'outdoors') THEN area
        WHEN surface_type = 'floor' AND (
          outside_boundary_condition_code = 3
          OR LOWER(outside_boundary_condition) IN (
            'ground',
            'groundfcfactormethod',
            'groundslabpreprocessoraverage',
            'groundslabpreprocessorcore',
            'groundslabpreprocessorperimeter',
            'groundbasementpreprocessoraveragefloor',
            'groundbasementpreprocessoraveragewall',
            'groundbasementpreprocessorlowerwall',
            'groundbasementpreprocessorupperwall',
            'foundation'
          )
        ) THEN area
        ELSE 0
      END
    ) * 1e4) / 1e4) AS external_surface_area

  FROM surfaces
  GROUP BY model_id, zone_name
),
all_zones AS (
  SELECT
    cz.model_id,
    cz.instance_name AS zone_name
  FROM ep.comp_zone cz
)
SELECT
  z.model_id,
  z.zone_name,
  bw.instance_name AS building_weather_instance_name,
  COALESCE(e.has_ext_wall, 0)::int AS has_ext_wall,
  COALESCE(e.has_ext_roof, 0)::int AS has_ext_roof,
  COALESCE(e.has_ground_contact, 0)::int AS has_ground_contact,
  COALESCE(e.n_external_surfaces, 0)::int AS n_external_surfaces,
  COALESCE(e.external_surface_area, 0)::double precision AS external_surface_area

FROM all_zones z
JOIN ep.building_weather bw
  ON bw.model_id = z.model_id
LEFT JOIN zone_external e
  ON e.model_id = z.model_id
 AND e.zone_name = z.zone_name
ORDER BY z.model_id, z.zone_name;
