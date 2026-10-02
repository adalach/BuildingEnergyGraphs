-- ============================================================================
-- edge_zones_model_weather.sql
--   • Lists, per model, the zones with at least one "external" boundary surface, with
--     exposure flags and areas; edge_zones_env_weather uses it as its exposure filter:
--     - external wall:   surface_type=Wall   AND outside_boundary_condition_code=1 (Outdoors)
--     - roof/ceiling:    surface_type=Roof/Ceiling AND outside_boundary_condition_code=1 (Outdoors)
--     - ground-contact:  surface_type=Floor  AND outside_boundary_condition_code=3 (Ground)
--   • Uses ep.buildingsurface_features for classification and area
-- ============================================================================

DROP VIEW IF EXISTS ep.edge_zones_model_weather CASCADE;

CREATE VIEW ep.edge_zones_model_weather AS
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
eligible AS (
  SELECT *
  FROM zone_external
  WHERE has_ext_wall = 1
     OR has_ext_roof = 1
     OR has_ground_contact = 1
)
SELECT
  e.model_id,
  e.zone_name,
  e.has_ext_wall,
  e.has_ext_roof,
  e.has_ground_contact,
  e.n_external_surfaces,
  e.external_surface_area
FROM eligible e
ORDER BY e.model_id, e.zone_name;
