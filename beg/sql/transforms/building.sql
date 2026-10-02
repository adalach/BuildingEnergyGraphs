-- ============================================================================
-- building.sql
--   • Creates a model-level "building" view starting from ep.ep_building
--   • Enriches each model_id with lighting, people, and outdoor air parameters
--   • Parses vintage + primary_function from ep.ep_people.instance_name
--   • Adds dynamic one-hot columns for vintage_* and primary_function_*
--   • Adds optional building-level shading proxy aggregates
--
-- Notes:
--   • Refers to schema `ep`, as every transform does; the pipeline rewrites it to
--     the target schema.
-- ============================================================================

DO $$
DECLARE
  vintage_onehot_sql text := '';
  function_onehot_sql text := '';
  sql_stmt text;
BEGIN
  /*
    1) Build dynamic one-hot columns from whatever categories exist right now.
       We derive categories from ep.ep_people.instance_name formatted like:
         "pre_1980::Retail::Retail_People"
       -> vintage = 'pre_1980'
       -> primary_function = 'Retail'
  */

  -- Dynamic one-hot for vintage
  SELECT
    COALESCE(
      string_agg(
        format(
          'CASE WHEN p.vintage = %L THEN 1 ELSE 0 END AS %I',
          v.vintage,
          'vintage_' || regexp_replace(lower(v.vintage), '[^a-z0-9]+', '_', 'g')
        ),
        E',\n        '
      ),
      ''
    )
  INTO vintage_onehot_sql
  FROM (
    SELECT DISTINCT split_part(instance_name, '::', 1) AS vintage
    FROM ep.ep_people
    WHERE instance_name LIKE '%::%::%'
      AND split_part(instance_name, '::', 1) IS NOT NULL
      AND btrim(split_part(instance_name, '::', 1)) <> ''
  ) v;

  -- Dynamic one-hot for primary_function
  SELECT
    COALESCE(
      string_agg(
        format(
          'CASE WHEN p.primary_function = %L THEN 1 ELSE 0 END AS %I',
          f.primary_function,
          'primary_function_' || regexp_replace(lower(f.primary_function), '[^a-z0-9]+', '_', 'g')
        ),
        E',\n        '
      ),
      ''
    )
  INTO function_onehot_sql
  FROM (
    SELECT DISTINCT split_part(instance_name, '::', 2) AS primary_function
    FROM ep.ep_people
    WHERE instance_name LIKE '%::%::%'
      AND split_part(instance_name, '::', 2) IS NOT NULL
      AND btrim(split_part(instance_name, '::', 2)) <> ''
  ) f;

  -- Drop the existing view
  EXECUTE 'DROP VIEW IF EXISTS ep.building CASCADE';

  /*
    2) Create the building view.
       Requirements:
         - keep instance_name + model_id (same value here, but both columns included)
         - ep_lights: return_air_fraction, fraction_radiant, fraction_visible (by model_id)
         - ep_people: people_per_floor_area (by model_id)
         - ep_designspecification_outdoorair:
              outdoor_air_flow_per_person,
              outdoor_air_flow_per_zone_floor_area,
              outdoor_air_flow_air_changes_per_hour
         - parse ep_people.instance_name into vintage + primary_function
         - then one-hot encode both (dynamic columns)
  */

  sql_stmt := format($SQL$
    CREATE VIEW ep.building AS
    WITH
    building_base AS (
      SELECT
        b.id,
        b.model_id,
        b.instance_name
      FROM ep.ep_building b
    ),

    lights AS (
      SELECT
        l.model_id,
        MAX(l.return_air_fraction) AS return_air_fraction,
        MAX(l.fraction_radiant)    AS fraction_radiant,
        MAX(l.fraction_visible)    AS fraction_visible
      FROM ep.ep_lights l
      GROUP BY l.model_id
    ),

    people_raw AS (
      SELECT
        p.model_id,
        MAX(p.people_per_floor_area) AS people_per_floor_area,
        -- choose any representative instance_name per model_id (usually 1 row anyway)
        MAX(p.instance_name) AS people_instance_name
      FROM ep.ep_people p
      GROUP BY p.model_id
    ),

    people_parsed AS (
      SELECT
        pr.model_id,
        pr.people_per_floor_area,
        pr.people_instance_name,
        NULLIF(btrim(split_part(pr.people_instance_name, '::', 1)), '') AS vintage,
        NULLIF(btrim(split_part(pr.people_instance_name, '::', 2)), '') AS primary_function
      FROM people_raw pr
    ),

    outdoor_air AS (
      SELECT
        o.model_id,
        MAX(o.outdoor_air_flow_per_person)            AS outdoor_air_flow_per_person,
        MAX(o.outdoor_air_flow_per_zone_floor_area)   AS outdoor_air_flow_per_zone_floor_area,
        MAX(o.outdoor_air_flow_air_changes_per_hour)  AS outdoor_air_flow_air_changes_per_hour
      FROM ep.ep_designspecification_outdoorair o
      GROUP BY o.model_id
    ),

    building_surface_geom AS (
      SELECT
        b.model_id,
        b.instance_name AS surface_name,
        COALESCE(b.area, 0)::double precision AS area
      FROM ep.buildingsurface_features b
      WHERE
        b.outside_boundary_condition_code = 1
        AND LOWER(b.surface_type) = 'wall'
    ),

    building_shading AS (
      SELECT
        g.model_id,
        g.area,
        sm.shade_count,
        sm.ray_count,
        sm.min_dist,
        sm.mean_dist,
        sm.p90_dist,
        sm.ray_dir_mean_sin,
        sm.ray_dir_mean_cos,
        sm.ray_elevation_mean_rad,
        sm.ray_elevation_mean_sin,
        sm.ray_elevation_mean_cos
      FROM building_surface_geom g
      LEFT JOIN ep.surface_shading_metrics_view sm
        ON sm.model_id = g.model_id
       AND sm.buildingsurface_id = g.surface_name
    ),

    building_shading_agg AS (
      SELECT
        model_id,
        SUM(COALESCE(shade_count, 0))::int AS shade_count,
        SUM(COALESCE(ray_count, 0))::int AS shade_ray_count,
        COALESCE(MIN(min_dist), 0) AS shade_min_dist,
        SUM(area * COALESCE(mean_dist, 0)) / NULLIF(SUM(area), 0) AS shade_mean_dist,
        SUM(area * COALESCE(p90_dist, 0)) / NULLIF(SUM(area), 0) AS shade_p90_dist,
        SUM(area * COALESCE(ray_dir_mean_sin, 0)) / NULLIF(SUM(area), 0) AS shade_ray_dir_mean_sin,
        SUM(area * COALESCE(ray_dir_mean_cos, 0)) / NULLIF(SUM(area), 0) AS shade_ray_dir_mean_cos,
        SUM(area * COALESCE(ray_elevation_mean_rad, 0)) / NULLIF(SUM(area), 0) AS shade_ray_elevation_mean_rad,
        SUM(area * COALESCE(ray_elevation_mean_sin, 0)) / NULLIF(SUM(area), 0) AS shade_ray_elevation_mean_sin,
        SUM(area * COALESCE(ray_elevation_mean_cos, 0)) / NULLIF(SUM(area), 0) AS shade_ray_elevation_mean_cos
      FROM building_shading
      GROUP BY model_id
    )

    SELECT
      bb.id,
      bb.model_id,
      bb.instance_name,

      -- Lights
      COALESCE(li.return_air_fraction, 0) AS return_air_fraction,
      COALESCE(li.fraction_radiant, 0)    AS fraction_radiant,
      COALESCE(li.fraction_visible, 0)    AS fraction_visible,

      -- People
      COALESCE(p.people_per_floor_area, 0) AS people_per_floor_area,

      -- Outdoor air
      COALESCE(oa.outdoor_air_flow_per_person, 0)           AS outdoor_air_flow_per_person,
      COALESCE(oa.outdoor_air_flow_per_zone_floor_area, 0)  AS outdoor_air_flow_per_zone_floor_area,
      COALESCE(oa.outdoor_air_flow_air_changes_per_hour, 0) AS outdoor_air_flow_air_changes_per_hour,

      -- Optional shading proxy aggregates
      COALESCE(bs.shade_count, 0) AS shade_count,
      COALESCE(bs.shade_ray_count, 0) AS shade_ray_count,
      COALESCE(bs.shade_min_dist, 0) AS shade_min_dist,
      COALESCE(bs.shade_mean_dist, 0) AS shade_mean_dist,
      COALESCE(bs.shade_p90_dist, 0) AS shade_p90_dist,
      COALESCE(bs.shade_ray_dir_mean_sin, 0) AS shade_ray_dir_mean_sin,
      COALESCE(bs.shade_ray_dir_mean_cos, 0) AS shade_ray_dir_mean_cos,
      COALESCE(bs.shade_ray_elevation_mean_rad, 0) AS shade_ray_elevation_mean_rad,
      COALESCE(bs.shade_ray_elevation_mean_sin, 0) AS shade_ray_elevation_mean_sin,
      COALESCE(bs.shade_ray_elevation_mean_cos, 0) AS shade_ray_elevation_mean_cos,

      -- Parsed categoricals
      p.vintage,
      p.primary_function
      %s
      %s

    FROM building_base bb
    LEFT JOIN lights li
      ON li.model_id = bb.model_id
    LEFT JOIN people_parsed p
      ON p.model_id = bb.model_id
    LEFT JOIN outdoor_air oa
      ON oa.model_id = bb.model_id
    LEFT JOIN building_shading_agg bs
      ON bs.model_id = bb.model_id
    ;
  $SQL$,
    CASE
      WHEN vintage_onehot_sql IS NOT NULL AND btrim(vintage_onehot_sql) <> ''
      THEN E',\n\n      -- One-hot: vintage\n      ' || vintage_onehot_sql
      ELSE ''
    END,
    CASE
      WHEN function_onehot_sql IS NOT NULL AND btrim(function_onehot_sql) <> ''
      THEN E',\n\n      -- One-hot: primary_function\n      ' || function_onehot_sql
      ELSE ''
    END
  );

  EXECUTE sql_stmt;
END $$;
