-- ============================================================================
-- comp_element.sql (Component Elements with Construction Properties)
--   • Creates a view that combines building surface features with construction material properties
--   • Uses suffix-aware chaining (looks for latest buildingsurface_* view)
--   • One-hot encodes surface_type (is_wall, is_ceiling, is_floor, is_roof)
--   • One-hot encodes outside_boundary_condition
--     (is_outdoors, is_surface, is_zone, is_ground, is_adiabatic)
--   • Includes boolean sun_exposure and wind_exposure
--   • Includes area from vertices calculation
--   • Includes optional shading-proxy metrics derived from nearby shading geometry
--   • Joins with construction_material_totals to get material properties:
--     - total_thickness
--     - r_value_layers (thermal resistance)
--     - u_value_total (thermal transmittance)
--     - solar_transmittance, visible_transmittance (standardized names)
--     - front_emissivity, back_emissivity (standardized names)
--     - thermal_absorptance, solar_absorptance, visible_absorptance (standardized names)
--     - density, specific_heat (standardized names)
--     - outside_roughness_category
--   • Standardized feature names, shared with comp_aperture and comp_base_element
-- ============================================================================

DO $$
DECLARE
  src_view   text;   -- table_name (no schema)
  out_view   text;   -- output view name: comp_element
  has_chain  boolean;

  -- Column existence flags for source view
  has_area   boolean;
  has_surface_type_text boolean;
  has_surface_type_onehot boolean;
  has_obc_onehot boolean;
  has_obc_text boolean;
  has_obc_code boolean;
  has_sun_exposed boolean;
  has_sun_exposure_onehot boolean;
  has_wind_exposed boolean;
  has_wind_exposure_onehot boolean;

  -- Specific one-hot column flags for surface_type
  has_surface_wall boolean;
  has_surface_floor boolean;
  has_surface_roof boolean;
  has_surface_ceiling boolean;

  -- Specific one-hot column flags for outside_boundary_condition
  has_obc_outdoors boolean;
  has_obc_surface boolean;
  has_obc_zone boolean;
  has_obc_ground boolean;
  has_obc_adiabatic boolean;

  -- SQL fragments for one-hot encoded columns
  surface_wall_sql text;
  surface_floor_sql text;
  surface_roof_sql text;
  surface_ceiling_sql text;

  obc_outdoors_sql text;
  obc_surface_sql text;
  obc_zone_sql text;
  obc_ground_sql text;
  obc_adiabatic_sql text;

  sun_exposure_sql text;
  wind_exposure_sql text;
  area_sql text;

  sql_stmt text;
BEGIN
  -- Set output view name
  out_view := 'comp_element';

  -- Find the "latest" buildingsurface_* view by underscore depth
  -- This allows chaining: buildingsurface_features -> buildingsurface_features_vertices -> etc.
  SELECT v.table_name
  INTO src_view
  FROM information_schema.views v
  WHERE v.table_schema='ep'
    AND v.table_name LIKE 'buildingsurface%'
    AND v.table_name NOT LIKE 'edge\_%'
  ORDER BY
    (char_length(v.table_name) - char_length(replace(v.table_name, '_', ''))) DESC,
    char_length(v.table_name) DESC,
    v.table_name DESC
  LIMIT 1;

  has_chain := src_view IS NOT NULL;

  IF NOT has_chain THEN
    src_view := 'ep_buildingsurface_detailed';
    RAISE NOTICE 'No buildingsurface_* views found, using base table: %', src_view;
  ELSE
    RAISE NOTICE 'Using chained view: %', src_view;
  END IF;

  -- Check for area column
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='area'
  ) INTO has_area;

  -- Check for surface_type text column (preferred fallback)
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type'
  ) INTO has_surface_type_text;

  -- Check for one-hot encoded columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^surface_type_'
  ) INTO has_surface_type_onehot;

  -- Check for outside boundary condition columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition'
  ) INTO has_obc_text;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_code'
  ) INTO has_obc_code;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^outside_boundary_condition_'
  ) INTO has_obc_onehot;

  -- Sun/wind exposure fallback columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='sun_exposed'
  ) INTO has_sun_exposed;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='wind_exposed'
  ) INTO has_wind_exposed;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^sun_exposure_'
  ) INTO has_sun_exposure_onehot;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^wind_exposure_'
  ) INTO has_wind_exposure_onehot;

  -- Check specific surface_type one-hot columns
  IF has_surface_type_onehot THEN
    SELECT
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_wall'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_floor'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_roof'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_ceiling')
    INTO has_surface_wall, has_surface_floor, has_surface_roof, has_surface_ceiling;
  END IF;

  -- Check specific outside_boundary_condition one-hot columns
  IF has_obc_onehot THEN
    SELECT
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_outdoors'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_surface'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_zone'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_ground'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition_adiabatic')
    INTO has_obc_outdoors, has_obc_surface, has_obc_zone, has_obc_ground, has_obc_adiabatic;
  END IF;

  -- Build SQL fragments for surface_type one-hot columns
  IF has_surface_wall THEN
    surface_wall_sql := 'COALESCE(b.surface_type_wall, 0)';
  ELSIF has_surface_type_text THEN
    surface_wall_sql := 'CASE WHEN LOWER(b.surface_type) = ''wall'' THEN 1 ELSE 0 END';
  ELSE
    surface_wall_sql := '0';
  END IF;

  IF has_surface_floor THEN
    surface_floor_sql := 'COALESCE(b.surface_type_floor, 0)';
  ELSIF has_surface_type_text THEN
    surface_floor_sql := 'CASE WHEN LOWER(b.surface_type) = ''floor'' THEN 1 ELSE 0 END';
  ELSE
    surface_floor_sql := '0';
  END IF;

  IF has_surface_roof THEN
    surface_roof_sql := 'COALESCE(b.surface_type_roof, 0)';
  ELSIF has_surface_type_text THEN
    surface_roof_sql := 'CASE WHEN LOWER(b.surface_type) IN (''roof'', ''roofceiling'', ''roof/ceiling'') THEN 1 ELSE 0 END';
  ELSE
    surface_roof_sql := '0';
  END IF;

  IF has_surface_ceiling THEN
    surface_ceiling_sql := 'COALESCE(b.surface_type_ceiling, 0)';
  ELSIF has_surface_type_text THEN
    surface_ceiling_sql := 'CASE WHEN LOWER(b.surface_type) = ''ceiling'' THEN 1 ELSE 0 END';
  ELSE
    surface_ceiling_sql := '0';
  END IF;

  -- Build SQL fragments for outside_boundary_condition one-hot columns
  IF has_obc_outdoors THEN
    obc_outdoors_sql := 'COALESCE(b.outside_boundary_condition_outdoors, 0)';
  ELSIF has_obc_code THEN
    obc_outdoors_sql := 'CASE WHEN b.outside_boundary_condition_code = 1 THEN 1 ELSE 0 END';
  ELSIF has_obc_text THEN
    obc_outdoors_sql := 'CASE WHEN LOWER(b.outside_boundary_condition) = ''outdoors'' THEN 1 ELSE 0 END';
  ELSE
    obc_outdoors_sql := '0';
  END IF;

  IF has_obc_surface THEN
    obc_surface_sql := 'COALESCE(b.outside_boundary_condition_surface, 0)';
  ELSIF has_obc_text THEN
    obc_surface_sql := 'CASE WHEN LOWER(b.outside_boundary_condition) = ''surface'' THEN 1 ELSE 0 END';
  ELSIF has_obc_code THEN
    -- Code=2 lumps 'Surface' + 'Zone' in buildingsurface_features; use as a coarse proxy.
    obc_surface_sql := 'CASE WHEN b.outside_boundary_condition_code = 2 THEN 1 ELSE 0 END';
  ELSE
    obc_surface_sql := '0';
  END IF;

  IF has_obc_zone THEN
    obc_zone_sql := 'COALESCE(b.outside_boundary_condition_zone, 0)';
  ELSIF has_obc_text THEN
    obc_zone_sql := 'CASE WHEN LOWER(b.outside_boundary_condition) = ''zone'' THEN 1 ELSE 0 END';
  ELSE
    obc_zone_sql := '0';
  END IF;

  IF has_obc_ground THEN
    obc_ground_sql := 'COALESCE(b.outside_boundary_condition_ground, 0)';
  ELSIF has_obc_code THEN
    obc_ground_sql := 'CASE WHEN b.outside_boundary_condition_code = 3 THEN 1 ELSE 0 END';
  ELSIF has_obc_text THEN
    obc_ground_sql := $GROUND$
CASE
  WHEN LOWER(b.outside_boundary_condition) IN (
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
  ) THEN 1
  ELSE 0
END
$GROUND$;
  ELSE
    obc_ground_sql := '0';
  END IF;

  IF has_obc_adiabatic THEN
    obc_adiabatic_sql := 'COALESCE(b.outside_boundary_condition_adiabatic, 0)';
  ELSIF has_obc_code THEN
    obc_adiabatic_sql := 'CASE WHEN b.outside_boundary_condition_code = 5 THEN 1 ELSE 0 END';
  ELSIF has_obc_text THEN
    obc_adiabatic_sql := 'CASE WHEN LOWER(b.outside_boundary_condition) = ''adiabatic'' THEN 1 ELSE 0 END';
  ELSE
    obc_adiabatic_sql := '0';
  END IF;

  -- Build SQL for sun_exposure (boolean/int)
  IF has_sun_exposed THEN
    sun_exposure_sql := 'COALESCE(b.sun_exposed, 0)';
  ELSIF has_sun_exposure_onehot THEN
    sun_exposure_sql := 'COALESCE(b.sun_exposure_sunexposed, 0)';
  ELSE
    sun_exposure_sql := '0';
  END IF;

  -- Build SQL for wind_exposure (boolean/int)
  IF has_wind_exposed THEN
    wind_exposure_sql := 'COALESCE(b.wind_exposed, 0)';
  ELSIF has_wind_exposure_onehot THEN
    wind_exposure_sql := 'COALESCE(b.wind_exposure_windexposed, 0)';
  ELSE
    wind_exposure_sql := '0';
  END IF;

  -- Build SQL for area
  area_sql := CASE WHEN has_area THEN 'b.area::double precision' ELSE '0::double precision' END;

  -- Drop and recreate the comp_element view
  EXECUTE format('DROP VIEW IF EXISTS ep.%I CASCADE;', out_view);

  sql_stmt := format($SQL$
    CREATE VIEW ep.%I AS
    SELECT
      b.id,
      b.model_id,
      b.instance_name,
      b.construction_name,
      b.zone_name,
      b.space_name,

      -- Surface type one-hot encoding
      %s::int AS is_wall,
      %s::int AS is_floor,
      %s::int AS is_roof,
      %s::int AS is_ceiling,

      -- Outside boundary condition one-hot encoding
      %s::int AS is_outdoors,
      %s::int AS is_surface,
      %s::int AS is_zone,
      %s::int AS is_ground,
      %s::int AS is_adiabatic,

      -- Exposure booleans
      %s::int AS sun_exposure,
      %s::int AS wind_exposure,

      -- Geometric properties
      ROUND((%s) * 1e4) / 1e4 AS area,


      -- Construction material properties (from construction_material_totals)
      -- Using standardized names (matching comp_aperture and comp_base_element)
      c.total_thickness,
      c.r_value_layers,
      c.u_value_total,
      c.avg_solar_trans AS solar_transmittance,
      c.avg_visible_trans AS visible_transmittance,
      c.avg_front_emiss AS front_emissivity,
      c.avg_back_emiss AS back_emissivity,
      c.avg_thermal_absorptance AS thermal_absorptance,
      c.avg_solar_absorptance AS solar_absorptance,
      c.avg_visible_absorptance AS visible_absorptance,
      c.avg_density AS density,
      c.avg_specific_heat AS specific_heat,
      c.outside_roughness_category,

      -- Optional shading proxy features.
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

    FROM ep.%I AS b
    LEFT JOIN ep.construction_material_totals AS c
      ON c.model_id = b.model_id
      AND c.instance_name = b.construction_name
    LEFT JOIN ep.surface_shading_metrics_view AS sm
      ON sm.model_id = b.model_id
      AND sm.buildingsurface_id = b.instance_name;
  $SQL$,
    out_view,
    surface_wall_sql,
    surface_floor_sql,
    surface_roof_sql,
    surface_ceiling_sql,
    obc_outdoors_sql,
    obc_surface_sql,
    obc_zone_sql,
    obc_ground_sql,
    obc_adiabatic_sql,
    sun_exposure_sql,
    wind_exposure_sql,
    area_sql,
    src_view
  );

  EXECUTE sql_stmt;

  RAISE NOTICE 'Created view ep.% from ep.%', out_view, src_view;
  RAISE NOTICE '  Surface type one-hot: wall=%, floor=%, roof=%, ceiling=%',
    has_surface_wall, has_surface_floor, has_surface_roof, has_surface_ceiling;
  RAISE NOTICE '  Outside boundary condition one-hot: outdoors=%, surface=%, zone=%, ground=%, adiabatic=%',
    has_obc_outdoors, has_obc_surface, has_obc_zone, has_obc_ground, has_obc_adiabatic;
  RAISE NOTICE '  Exposure: sun=%, wind=%', has_sun_exposure_onehot, has_wind_exposure_onehot;
  RAISE NOTICE '  Area column: %', has_area;
  RAISE NOTICE '  Construction properties joined from construction_material_totals';
END$$;
