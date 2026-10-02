-- ============================================================================
-- comp_aperture.sql (Component Apertures - Windows, Doors, Skylights)
--   • Creates a view for fenestration surfaces (windows, doors, skylights)
--   • Calculates area from vertex coordinates (stored as columns)
--   • Joins with construction_material_totals for glazing properties
--   • Includes parent surface reference (building_surface_name)
--   • Provides optical and thermal properties specific to glazing systems
--   • Inherits optional shading-proxy metrics from the hosting surface
--   • Handles missing surface_type column (assumes Window if missing)
-- ============================================================================

DO $$
DECLARE
  has_surface_type boolean;
  source_relkind char;
  view_sql text;
BEGIN
  -- Check if surface_type column exists
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep'
      AND table_name='ep_fenestrationsurface_detailed'
      AND column_name='surface_type'
  ) INTO has_surface_type;

  -- Drop existing view
  EXECUTE 'DROP VIEW IF EXISTS ep.comp_aperture CASCADE';

  -- Build SQL with conditional surface_type handling
  IF has_surface_type THEN
    -- Use actual surface_type column
    view_sql := $SQL$
CREATE VIEW ep.comp_aperture AS
WITH
/* 1) Calculate area from vertices ----------------------------------------- */
fenestration_with_area AS (
  SELECT
    f.id,
    f.model_id,
    f.instance_name,
    f.construction_name,
    f.building_surface_name,
    COALESCE(f.surface_type, 'Window') AS surface_type,

    -- Calculate area using cross product method for 4-vertex polygon
    -- Assumes vertices are ordered and form a planar quadrilateral
    0.5 * SQRT(
      POWER(
        (f.vertex_2_y_coordinate - f.vertex_1_y_coordinate) * (f.vertex_4_z_coordinate - f.vertex_1_z_coordinate) -
        (f.vertex_2_z_coordinate - f.vertex_1_z_coordinate) * (f.vertex_4_y_coordinate - f.vertex_1_y_coordinate) +
        (f.vertex_3_y_coordinate - f.vertex_2_y_coordinate) * (f.vertex_1_z_coordinate - f.vertex_2_z_coordinate) -
        (f.vertex_3_z_coordinate - f.vertex_2_z_coordinate) * (f.vertex_1_y_coordinate - f.vertex_2_y_coordinate)
      , 2) +
      POWER(
        (f.vertex_2_z_coordinate - f.vertex_1_z_coordinate) * (f.vertex_4_x_coordinate - f.vertex_1_x_coordinate) -
        (f.vertex_2_x_coordinate - f.vertex_1_x_coordinate) * (f.vertex_4_z_coordinate - f.vertex_1_z_coordinate) +
        (f.vertex_3_z_coordinate - f.vertex_2_z_coordinate) * (f.vertex_1_x_coordinate - f.vertex_2_x_coordinate) -
        (f.vertex_3_x_coordinate - f.vertex_2_x_coordinate) * (f.vertex_1_z_coordinate - f.vertex_2_z_coordinate)
      , 2) +
      POWER(
        (f.vertex_2_x_coordinate - f.vertex_1_x_coordinate) * (f.vertex_4_y_coordinate - f.vertex_1_y_coordinate) -
        (f.vertex_2_y_coordinate - f.vertex_1_y_coordinate) * (f.vertex_4_x_coordinate - f.vertex_1_x_coordinate) +
        (f.vertex_3_x_coordinate - f.vertex_2_x_coordinate) * (f.vertex_1_y_coordinate - f.vertex_2_y_coordinate) -
        (f.vertex_3_y_coordinate - f.vertex_2_y_coordinate) * (f.vertex_1_x_coordinate - f.vertex_2_x_coordinate)
      , 2)
    )::double precision AS area

  FROM ep.ep_fenestrationsurface_detailed f
)

/* 2) Main query with construction properties ------------------------------ */
SELECT
  fa.id,
  fa.model_id,
  fa.instance_name,
  fa.construction_name,
  fa.building_surface_name,
  fa.surface_type,

  -- Surface type indicators (fenestration types)
  CASE WHEN LOWER(fa.surface_type) = 'window' THEN 1 ELSE 0 END::int AS is_window,
  CASE WHEN LOWER(fa.surface_type) = 'door' THEN 1 ELSE 0 END::int AS is_door,
  CASE WHEN LOWER(fa.surface_type) IN ('glassdoor', 'glass door') THEN 1 ELSE 0 END::int AS is_glass_door,
  CASE WHEN LOWER(fa.surface_type) IN ('tubulardaylightdome', 'tubulardaylightdiffuser') THEN 1 ELSE 0 END::int AS is_tubular_daylight,

  -- Geometric properties
  ROUND(fa.area * 1e4) / 1e4 AS area,


  -- Construction/glazing properties (from construction_material_totals)
  -- These are particularly relevant for glazing systems
  c.total_thickness,
  c.r_value_layers,
  c.u_value_total,

  -- Optical properties (critical for windows/glazing)
  c.avg_solar_trans AS solar_transmittance,
  c.avg_visible_trans AS visible_transmittance,
  c.avg_front_emiss AS front_emissivity,
  c.avg_back_emiss AS back_emissivity,

  -- Absorptance properties
  c.avg_thermal_absorptance AS thermal_absorptance,
  c.avg_solar_absorptance AS solar_absorptance,
  c.avg_visible_absorptance AS visible_absorptance,

  -- Material properties (less critical for glazing but included for completeness)
  c.avg_density AS density,
  c.avg_specific_heat AS specific_heat,
  c.outside_roughness_category,

  -- Optional shading proxy features inherited from the host surface.
  COALESCE(sm.shade_count, 0) AS shade_count,
  COALESCE(sm.ray_count, 0) AS shade_ray_count,
  COALESCE(sm.min_dist, 0) AS shade_min_dist,
  COALESCE(sm.mean_dist, 0) AS shade_mean_dist,
  COALESCE(sm.p90_dist, 0) AS shade_p90_dist,
  COALESCE(sm.ray_dir_mean_sin, 0) AS shade_ray_dir_mean_sin,
  COALESCE(sm.ray_dir_mean_cos, 0) AS shade_ray_dir_mean_cos,
  COALESCE(sm.ray_elevation_mean_rad, 0) AS shade_ray_elevation_mean_rad,
  COALESCE(sm.ray_elevation_mean_sin, 0) AS shade_ray_elevation_mean_sin,
  COALESCE(sm.ray_elevation_mean_cos, 0) AS shade_ray_elevation_mean_cos

FROM fenestration_with_area fa
LEFT JOIN ep.construction_material_totals c
  ON c.model_id = fa.model_id
  AND c.instance_name = fa.construction_name
LEFT JOIN ep.surface_shading_metrics_view sm
  ON sm.model_id = fa.model_id
  AND sm.buildingsurface_id = fa.building_surface_name
$SQL$;
  ELSE
    -- surface_type column doesn't exist - assume all are Windows
    view_sql := $SQL$
CREATE VIEW ep.comp_aperture AS
WITH
/* 1) Calculate area from vertices ----------------------------------------- */
fenestration_with_area AS (
  SELECT
    f.id,
    f.model_id,
    f.instance_name,
    f.construction_name,
    f.building_surface_name,
    'Window'::text AS surface_type,

    -- Calculate area using cross product method for 4-vertex polygon
    -- Assumes vertices are ordered and form a planar quadrilateral
    0.5 * SQRT(
      POWER(
        (f.vertex_2_y_coordinate - f.vertex_1_y_coordinate) * (f.vertex_4_z_coordinate - f.vertex_1_z_coordinate) -
        (f.vertex_2_z_coordinate - f.vertex_1_z_coordinate) * (f.vertex_4_y_coordinate - f.vertex_1_y_coordinate) +
        (f.vertex_3_y_coordinate - f.vertex_2_y_coordinate) * (f.vertex_1_z_coordinate - f.vertex_2_z_coordinate) -
        (f.vertex_3_z_coordinate - f.vertex_2_z_coordinate) * (f.vertex_1_y_coordinate - f.vertex_2_y_coordinate)
      , 2) +
      POWER(
        (f.vertex_2_z_coordinate - f.vertex_1_z_coordinate) * (f.vertex_4_x_coordinate - f.vertex_1_x_coordinate) -
        (f.vertex_2_x_coordinate - f.vertex_1_x_coordinate) * (f.vertex_4_z_coordinate - f.vertex_1_z_coordinate) +
        (f.vertex_3_z_coordinate - f.vertex_2_z_coordinate) * (f.vertex_1_x_coordinate - f.vertex_2_x_coordinate) -
        (f.vertex_3_x_coordinate - f.vertex_2_x_coordinate) * (f.vertex_1_z_coordinate - f.vertex_2_z_coordinate)
      , 2) +
      POWER(
        (f.vertex_2_x_coordinate - f.vertex_1_x_coordinate) * (f.vertex_4_y_coordinate - f.vertex_1_y_coordinate) -
        (f.vertex_2_y_coordinate - f.vertex_1_y_coordinate) * (f.vertex_4_x_coordinate - f.vertex_1_x_coordinate) +
        (f.vertex_3_x_coordinate - f.vertex_2_x_coordinate) * (f.vertex_1_y_coordinate - f.vertex_2_y_coordinate) -
        (f.vertex_3_y_coordinate - f.vertex_2_y_coordinate) * (f.vertex_1_x_coordinate - f.vertex_2_x_coordinate)
      , 2)
    )::double precision AS area

  FROM ep.ep_fenestrationsurface_detailed f
)

/* 2) Main query with construction properties ------------------------------ */
SELECT
  fa.id,
  fa.model_id,
  fa.instance_name,
  fa.construction_name,
  fa.building_surface_name,
  fa.surface_type,

  -- Surface type indicators (all windows since column doesn't exist)
  1::int AS is_window,
  0::int AS is_door,
  0::int AS is_glass_door,
  0::int AS is_tubular_daylight,

  -- Geometric properties
  ROUND(fa.area * 1e4) / 1e4 AS area,


  -- Construction/glazing properties (from construction_material_totals)
  c.total_thickness,
  c.r_value_layers,
  c.u_value_total,

  -- Optical properties (critical for windows/glazing)
  c.avg_solar_trans AS solar_transmittance,
  c.avg_visible_trans AS visible_transmittance,
  c.avg_front_emiss AS front_emissivity,
  c.avg_back_emiss AS back_emissivity,

  -- Absorptance properties
  c.avg_thermal_absorptance AS thermal_absorptance,
  c.avg_solar_absorptance AS solar_absorptance,
  c.avg_visible_absorptance AS visible_absorptance,

  -- Material properties
  c.avg_density AS density,
  c.avg_specific_heat AS specific_heat,
  c.outside_roughness_category,

  -- Optional shading proxy features inherited from the host surface.
  COALESCE(sm.shade_count, 0) AS shade_count,
  COALESCE(sm.ray_count, 0) AS shade_ray_count,
  COALESCE(sm.min_dist, 0) AS shade_min_dist,
  COALESCE(sm.mean_dist, 0) AS shade_mean_dist,
  COALESCE(sm.p90_dist, 0) AS shade_p90_dist,
  COALESCE(sm.ray_dir_mean_sin, 0) AS shade_ray_dir_mean_sin,
  COALESCE(sm.ray_dir_mean_cos, 0) AS shade_ray_dir_mean_cos,
  COALESCE(sm.ray_elevation_mean_rad, 0) AS shade_ray_elevation_mean_rad,
  COALESCE(sm.ray_elevation_mean_sin, 0) AS shade_ray_elevation_mean_sin,
  COALESCE(sm.ray_elevation_mean_cos, 0) AS shade_ray_elevation_mean_cos

FROM fenestration_with_area fa
LEFT JOIN ep.construction_material_totals c
  ON c.model_id = fa.model_id
  AND c.instance_name = fa.construction_name
LEFT JOIN ep.surface_shading_metrics_view sm
  ON sm.model_id = fa.model_id
  AND sm.buildingsurface_id = fa.building_surface_name
$SQL$;
  END IF;

  -- Execute the appropriate SQL
  EXECUTE view_sql;

  -- Create indexes for better query performance only if source is indexable.
  -- In tables mode the pipeline may prepare `ep_fenestrationsurface_detailed` as a shim
  -- view inside `ep_transformed`; indexes on views are not supported.
  SELECT c.relkind
  INTO source_relkind
  FROM pg_catalog.pg_class c
  JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'ep'
    AND c.relname = 'ep_fenestrationsurface_detailed'
  LIMIT 1;

  IF source_relkind IN ('r', 'p', 'm') THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_comp_aperture_model_id ON ep.ep_fenestrationsurface_detailed(model_id)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_comp_aperture_building_surface ON ep.ep_fenestrationsurface_detailed(model_id, building_surface_name)';
  ELSE
    RAISE NOTICE
      'Skipping comp_aperture source indexes; relation ep.ep_fenestrationsurface_detailed is relkind=%',
      source_relkind;
  END IF;

  -- Log what was done
  IF has_surface_type THEN
    RAISE NOTICE 'Created view ep.comp_aperture with surface_type column';
  ELSE
    RAISE NOTICE 'Created view ep.comp_aperture WITHOUT surface_type column (assuming all are Windows)';
  END IF;
END$$;
