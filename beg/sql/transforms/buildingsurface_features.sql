DO $$
DECLARE
  src_view   text;   -- table_name (no schema)
  out_view   text;   -- table_name (no schema)
  has_chain  boolean;
  has_area   boolean; -- unused

  -- Geometry join flags
  has_id     boolean;
  has_name   boolean;

  -- Column existence flags
  has_surface_type_text boolean;
  has_surface_type_onehot boolean;
  has_sun_exposure_text boolean;
  has_sun_exposure_onehot boolean;
  has_wind_exposure_text boolean;
  has_wind_exposure_onehot boolean;
  has_obc_text boolean;  -- outside_boundary_condition
  has_obc_onehot boolean;

  -- Specific one-hot column flags
  has_obc_outdoors boolean;
  has_obc_surface boolean;
  has_obc_zone boolean;
  has_obc_ground boolean;
  has_obc_adiabatic boolean;

  has_surface_wall boolean;
  has_surface_floor boolean;
  has_surface_roof boolean;
  has_surface_ceiling boolean;

  -- SQL fragments
  surface_type_sql text;
  sun_exposed_sql text;
  wind_exposed_sql text;
  obc_sql text;
  obc_code_sql text;

  sql_stmt   text;
BEGIN
  -- Find the "latest" buildingsurface_* view by underscore depth
  SELECT v.table_name
  INTO src_view
  FROM information_schema.views v
  WHERE v.table_schema='ep'
    AND v.table_name ~ '^buildingsurface_'
    AND v.table_name !~ '^edge_'
    AND v.table_name !~ '_features$'
  ORDER BY
    (char_length(v.table_name) - char_length(replace(v.table_name, '_', ''))) DESC,
    char_length(v.table_name) DESC,
    v.table_name DESC
  LIMIT 1;

  has_chain := src_view IS NOT NULL;

  IF NOT has_chain THEN
    src_view := 'ep_buildingsurface_detailed';
  END IF;

  -- Always create a single target view name
  out_view := 'buildingsurface_features';

  -- Check for surface_type columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type'
  ) INTO has_surface_type_text;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^surface_type_'
  ) INTO has_surface_type_onehot;

  -- Check for sun_exposure columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='sun_exposure'
  ) INTO has_sun_exposure_text;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^sun_exposure_'
  ) INTO has_sun_exposure_onehot;

  -- Check for wind_exposure columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='wind_exposure'
  ) INTO has_wind_exposure_text;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^wind_exposure_'
  ) INTO has_wind_exposure_onehot;

  -- Check for outside_boundary_condition columns
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view AND column_name='outside_boundary_condition'
  ) INTO has_obc_text;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='ep' AND table_name=src_view
    AND column_name ~ '^outside_boundary_condition_'
  ) INTO has_obc_onehot;

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

  -- Check specific surface_type one-hot columns
  IF has_surface_type_onehot THEN
    SELECT
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_wall'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_floor'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_roof'),
      EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name=src_view AND column_name='surface_type_ceiling')
    INTO has_surface_wall, has_surface_floor, has_surface_roof, has_surface_ceiling;
  END IF;

  -- Build surface_type SQL (reconstruct from one-hot or pass through)
  IF has_surface_type_text THEN
    surface_type_sql := 'b.surface_type';
  ELSIF has_surface_type_onehot THEN
    surface_type_sql := 'CASE';
    IF has_surface_wall THEN
      surface_type_sql := surface_type_sql || E'\n  WHEN b.surface_type_wall = 1 THEN ''Wall''';
    END IF;
    IF has_surface_floor THEN
      surface_type_sql := surface_type_sql || E'\n  WHEN b.surface_type_floor = 1 THEN ''Floor''';
    END IF;
    IF has_surface_roof THEN
      surface_type_sql := surface_type_sql || E'\n  WHEN b.surface_type_roof = 1 THEN ''Roof''';
    END IF;
    IF has_surface_ceiling THEN
      surface_type_sql := surface_type_sql || E'\n  WHEN b.surface_type_ceiling = 1 THEN ''Ceiling''';
    END IF;
    surface_type_sql := surface_type_sql || E'\n  ELSE NULL\nEND';
  ELSE
    surface_type_sql := 'NULL';
  END IF;

  -- Build sun_exposed SQL
  IF has_sun_exposure_text THEN
    sun_exposed_sql := 'CASE b.sun_exposure WHEN ''SunExposed'' THEN 1 ELSE 0 END';
  ELSIF has_sun_exposure_onehot THEN
    sun_exposed_sql := 'COALESCE(b.sun_exposure_sunexposed, 0)';
  ELSE
    sun_exposed_sql := '0';
  END IF;

  -- Build wind_exposed SQL
  IF has_wind_exposure_text THEN
    wind_exposed_sql := 'CASE b.wind_exposure WHEN ''WindExposed'' THEN 1 ELSE 0 END';
  ELSIF has_wind_exposure_onehot THEN
    wind_exposed_sql := 'COALESCE(b.wind_exposure_windexposed, 0)';
  ELSE
    wind_exposed_sql := '0';
  END IF;

  -- Build outside_boundary_condition SQL (for display)
  IF has_obc_text THEN
    obc_sql := 'b.outside_boundary_condition';
  ELSIF has_obc_onehot THEN
    obc_sql := 'CASE';
    IF has_obc_outdoors THEN
      obc_sql := obc_sql || E'\n  WHEN b.outside_boundary_condition_outdoors = 1 THEN ''Outdoors''';
    END IF;
    IF has_obc_surface THEN
      obc_sql := obc_sql || E'\n  WHEN b.outside_boundary_condition_surface = 1 THEN ''Surface''';
    END IF;
    IF has_obc_zone THEN
      obc_sql := obc_sql || E'\n  WHEN b.outside_boundary_condition_zone = 1 THEN ''Zone''';
    END IF;
    IF has_obc_ground THEN
      obc_sql := obc_sql || E'\n  WHEN b.outside_boundary_condition_ground = 1 THEN ''Ground''';
    END IF;
    IF has_obc_adiabatic THEN
      obc_sql := obc_sql || E'\n  WHEN b.outside_boundary_condition_adiabatic = 1 THEN ''Adiabatic''';
    END IF;
    obc_sql := obc_sql || E'\n  ELSE NULL\nEND';
  ELSE
    obc_sql := 'NULL';
  END IF;

  -- Build outside_boundary_condition_code SQL (numeric categorization)
  IF has_obc_text THEN
    obc_code_sql := $OBC_CODE$
CASE
  WHEN b.outside_boundary_condition IN ('Outdoors') THEN 1
  WHEN b.outside_boundary_condition IN ('Surface','Zone') THEN 2
  WHEN b.outside_boundary_condition IN (
    'Ground','GroundFCfactorMethod',
    'GroundSlabPreprocessorAverage','GroundSlabPreprocessorCore','GroundSlabPreprocessorPerimeter',
    'GroundBasementPreprocessorAverageFloor','GroundBasementPreprocessorAverageWall',
    'GroundBasementPreprocessorLowerWall','GroundBasementPreprocessorUpperWall',
    'Foundation'
  ) THEN 3
  WHEN b.outside_boundary_condition IN ('OtherSideCoefficients','OtherSideConditionsModel') THEN 4
  WHEN b.outside_boundary_condition = 'Adiabatic' THEN 5
  ELSE 9
END
$OBC_CODE$;
  ELSIF has_obc_onehot THEN
    obc_code_sql := 'CASE';
    IF has_obc_outdoors THEN
      obc_code_sql := obc_code_sql || E'\n  WHEN b.outside_boundary_condition_outdoors = 1 THEN 1';
    END IF;
    -- Category 2: Surface or Zone (if either exists)
    IF has_obc_surface OR has_obc_zone THEN
      obc_code_sql := obc_code_sql || E'\n  WHEN ';
      IF has_obc_surface AND has_obc_zone THEN
        obc_code_sql := obc_code_sql || 'b.outside_boundary_condition_surface = 1 OR b.outside_boundary_condition_zone = 1';
      ELSIF has_obc_surface THEN
        obc_code_sql := obc_code_sql || 'b.outside_boundary_condition_surface = 1';
      ELSE
        obc_code_sql := obc_code_sql || 'b.outside_boundary_condition_zone = 1';
      END IF;
      obc_code_sql := obc_code_sql || ' THEN 2';
    END IF;
    IF has_obc_ground THEN
      obc_code_sql := obc_code_sql || E'\n  WHEN b.outside_boundary_condition_ground = 1 THEN 3';
    END IF;
    IF has_obc_adiabatic THEN
      obc_code_sql := obc_code_sql || E'\n  WHEN b.outside_boundary_condition_adiabatic = 1 THEN 5';
    END IF;
    obc_code_sql := obc_code_sql || E'\n  ELSE 9\nEND';
  ELSE
    obc_code_sql := '9';
  END IF;

  -- Detect presence of id / instance_name in chosen source for area join
  SELECT EXISTS (
           SELECT 1 FROM information_schema.columns
           WHERE table_schema='ep' AND table_name=src_view AND column_name='id'
         )
    INTO has_id;

  SELECT EXISTS (
           SELECT 1 FROM information_schema.columns
           WHERE table_schema='ep' AND table_name=src_view AND column_name='instance_name'
         )
    INTO has_name;

  IF NOT has_id AND NOT has_name THEN
    RAISE EXCEPTION 'Neither id nor instance_name exists in ep.% (needed for area join)', src_view;
  END IF;

  -- Recreate target view
  EXECUTE format('DROP VIEW IF EXISTS ep.%I CASCADE;', out_view);

  -- Build geometry + feature view in a single step
  IF has_id THEN
    -- Fast path: source has id → join areas directly on b.id
    sql_stmt := format($SQL$
      CREATE VIEW ep.%I AS
      WITH ordered AS (
        SELECT
          v.parent_id,
          v.position,
          v.vertex_x_coordinate::double precision AS x,
          v.vertex_y_coordinate::double precision AS y,
          v.vertex_z_coordinate::double precision AS z
        FROM ep.ep_buildingsurface_detailed_vertices v
        ORDER BY v.parent_id, v.position
      ),
      edges AS (
        SELECT
          o.parent_id,
          o.x, o.y, o.z,
          COALESCE(LEAD(o.x) OVER w, FIRST_VALUE(o.x) OVER w) AS nx,
          COALESCE(LEAD(o.y) OVER w, FIRST_VALUE(o.y) OVER w) AS ny,
          COALESCE(LEAD(o.z) OVER w, FIRST_VALUE(o.z) OVER w) AS nz
        FROM ordered o
        WINDOW w AS (PARTITION BY o.parent_id ORDER BY o.position)
      ),
      crosssum AS (
        SELECT
          parent_id,
          SUM( (y*nz - z*ny) ) AS sx,
          SUM( (z*nx - x*nz) ) AS sy,
          SUM( (x*ny - y*nx) ) AS sz
        FROM edges
        GROUP BY parent_id
      ),
      areas AS (
        SELECT parent_id, 0.5 * sqrt(sx*sx + sy*sy + sz*sz) AS area
        FROM crosssum
      )
      SELECT
        b.id,
        b.model_id,
        b.instance_name,
        %s AS surface_type,
        b.construction_name,
        b.zone_name,
        b.space_name,

        %s::int AS sun_exposed,
        %s::int AS wind_exposed,

        %s AS outside_boundary_condition,
        b.outside_boundary_condition_object,

        %s::int AS outside_boundary_condition_code,
        COALESCE(a.area, 0)::double precision AS area
      FROM ep.%I AS b
      LEFT JOIN areas a
        ON a.parent_id = b.id;
    $SQL$,
      out_view,
      surface_type_sql,
      sun_exposed_sql,
      wind_exposed_sql,
      obc_sql,
      obc_code_sql,
      src_view
    );
  ELSE
    -- Fallback: source lacks id → hop via base table by instance_name to recover base id
    sql_stmt := format($SQL$
      CREATE VIEW ep.%I AS
      WITH ordered AS (
        SELECT
          v.parent_id,
          v.position,
          v.vertex_x_coordinate::double precision AS x,
          v.vertex_y_coordinate::double precision AS y,
          v.vertex_z_coordinate::double precision AS z
        FROM ep.ep_buildingsurface_detailed_vertices v
        ORDER BY v.parent_id, v.position
      ),
      edges AS (
        SELECT
          o.parent_id,
          o.x, o.y, o.z,
          COALESCE(LEAD(o.x) OVER w, FIRST_VALUE(o.x) OVER w) AS nx,
          COALESCE(LEAD(o.y) OVER w, FIRST_VALUE(o.y) OVER w) AS ny,
          COALESCE(LEAD(o.z) OVER w, FIRST_VALUE(o.z) OVER w) AS nz
        FROM ordered o
        WINDOW w AS (PARTITION BY o.parent_id ORDER BY o.position)
      ),
      crosssum AS (
        SELECT
          parent_id,
          SUM( (y*nz - z*ny) ) AS sx,
          SUM( (z*nx - x*nz) ) AS sy,
          SUM( (x*ny - y*nx) ) AS sz
        FROM edges
        GROUP BY parent_id
      ),
      areas AS (
        SELECT parent_id, 0.5 * sqrt(sx*sx + sy*sy + sz*sz) AS area
        FROM crosssum
      )
      SELECT
        b.id,
        b.model_id,
        b.instance_name,
        %s AS surface_type,
        b.construction_name,
        b.zone_name,
        b.space_name,

        %s::int AS sun_exposed,
        %s::int AS wind_exposed,

        %s AS outside_boundary_condition,
        b.outside_boundary_condition_object,

        %s::int AS outside_boundary_condition_code,
        COALESCE(a.area, 0)::double precision AS area
      FROM ep.%I AS b
      JOIN ep.ep_buildingsurface_detailed d
        ON d.instance_name = b.instance_name
      LEFT JOIN areas a
        ON a.parent_id = d.id;
    $SQL$,
      out_view,
      surface_type_sql,
      sun_exposed_sql,
      wind_exposed_sql,
      obc_sql,
      obc_code_sql,
      src_view
    );
  END IF;

  EXECUTE sql_stmt;

  RAISE NOTICE 'Created view ep.% from ep.%', out_view, src_view;
  RAISE NOTICE '  surface_type: %', CASE WHEN has_surface_type_text THEN 'text' WHEN has_surface_type_onehot THEN 'one-hot' ELSE 'missing' END;
  RAISE NOTICE '  sun_exposure: %', CASE WHEN has_sun_exposure_text THEN 'text' WHEN has_sun_exposure_onehot THEN 'one-hot' ELSE 'missing' END;
  RAISE NOTICE '  wind_exposure: %', CASE WHEN has_wind_exposure_text THEN 'text' WHEN has_wind_exposure_onehot THEN 'one-hot' ELSE 'missing' END;
  RAISE NOTICE '  outside_boundary_condition: %', CASE WHEN has_obc_text THEN 'text' WHEN has_obc_onehot THEN 'one-hot' ELSE 'missing' END;
END$$;
