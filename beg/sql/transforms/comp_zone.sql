-- ============================================================================
-- comp_zone.sql (Component Zones with Systems, Schedules, and Loads)
--   • Creates an enriched zone view from minimal ep_zone table
--   • Calculates properties from building surface geometry and related tables
--   • Includes ceiling_height calculated from wall heights
--   • Includes floor_level (ground=0) from z-coordinates
--   • Joins with wateruse_equipment for peak_flow_rate
--   • Joins with zoneinfiltration for flow rates and schedules
--   • Adds optional zone-level shading proxy aggregates
-- ============================================================================

DO $$
DECLARE
  -- Always keep identity + core geometric columns, even if constant.
  keep_cols text[] := ARRAY[
    'id', 'model_id', 'instance_name',
    'floor_area', 'ceiling_height', 'floor_level', 'is_ground_floor', 'is_top_floor'
  ];
  col record;
  distinct_count bigint;
  select_list text := '';
  base_select text;
  has_infiltration_schedule_name boolean;
  has_infiltration_schedule_apartmentmidrise boolean;
  has_infiltration_schedule_midrise boolean;
  has_infiltration_schedule_officemedium boolean;
  infiltration_schedule_sql text;
  schedule_case_sql text;
BEGIN
  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='ep'
      AND table_name='ep_zoneinfiltration_designflowrate'
      AND column_name='schedule_name'
  ) INTO has_infiltration_schedule_name;

  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='ep'
      AND table_name='ep_zoneinfiltration_designflowrate'
      AND column_name='schedule_name_apartmentmidrise_inf_apt_sch'
  ) INTO has_infiltration_schedule_apartmentmidrise;

  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='ep'
      AND table_name='ep_zoneinfiltration_designflowrate'
      AND column_name='schedule_name_midriseapartment_infil'
  ) INTO has_infiltration_schedule_midrise;

  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='ep'
      AND table_name='ep_zoneinfiltration_designflowrate'
      AND column_name='schedule_name_officemedium_infil_sch_pnnl'
  ) INTO has_infiltration_schedule_officemedium;

  IF has_infiltration_schedule_name THEN
    infiltration_schedule_sql := 'MAX(i.schedule_name)';
  ELSE
    schedule_case_sql := '';
    IF has_infiltration_schedule_apartmentmidrise THEN
      schedule_case_sql := schedule_case_sql ||
        'CASE WHEN MAX(i.schedule_name_apartmentmidrise_inf_apt_sch) = 1 ' ||
        'THEN ''ApartmentMidrise_Inf_Apt_Sch'' END, ';
    END IF;
    IF has_infiltration_schedule_midrise THEN
      schedule_case_sql := schedule_case_sql ||
        'CASE WHEN MAX(i.schedule_name_midriseapartment_infil) = 1 ' ||
        'THEN ''MidriseApartment_Infil'' END, ';
    END IF;
    IF has_infiltration_schedule_officemedium THEN
      schedule_case_sql := schedule_case_sql ||
        'CASE WHEN MAX(i.schedule_name_officemedium_infil_sch_pnnl) = 1 ' ||
        'THEN ''OfficeMedium_Infil_Sch_PNNL'' END, ';
    END IF;
    schedule_case_sql := regexp_replace(schedule_case_sql, E',\\s*$', '');
    IF schedule_case_sql IS NULL OR btrim(schedule_case_sql) = '' THEN
      infiltration_schedule_sql := 'NULL::text';
    ELSE
      infiltration_schedule_sql := format('COALESCE(%s)', schedule_case_sql);
    END IF;
  END IF;

  EXECUTE 'DROP VIEW IF EXISTS ep.comp_zone CASCADE';

  -- 1) Create a full (unpruned) view first ----------------------------------
  EXECUTE format($SQL$
    CREATE VIEW ep.comp_zone AS
    WITH
    /* 1) Base zone (only has id, model_id, instance_name) -------------------- */
    zone_base AS (
      SELECT
        z.id,
        z.model_id,
        z.instance_name
      FROM ep.ep_zone z
    ),

    /* 2) Geometry: floor area from floor surfaces + min z -------------------- */
    zone_floor_area AS (
      SELECT
        b.model_id,
        b.zone_name,
        SUM(CASE WHEN LOWER(b.surface_type) = 'floor' THEN b.area::double precision ELSE 0::double precision END) AS floor_area
      FROM ep.buildingsurface_features b
      JOIN zone_base zb
        ON b.model_id = zb.model_id
       AND b.zone_name = zb.instance_name
      WHERE b.zone_name IS NOT NULL
      GROUP BY b.model_id, b.zone_name
    ),
    zone_min_z AS (
      SELECT
        b.model_id,
        b.zone_name,
        MIN(v.vertex_z_coordinate::double precision) AS min_z_coordinate
      FROM ep.buildingsurface_features b
      JOIN zone_base zb
        ON b.model_id = zb.model_id
       AND b.zone_name = zb.instance_name
      JOIN ep.ep_buildingsurface_detailed_vertices v ON v.parent_id = b.id
      WHERE b.zone_name IS NOT NULL
      GROUP BY b.model_id, b.zone_name
    ),
    zone_geometry AS (
      SELECT
        k.model_id,
        k.zone_name,
        COALESCE(zfa.floor_area, 0) AS floor_area,
        zmz.min_z_coordinate
      FROM (
        SELECT model_id, zone_name FROM zone_floor_area
        UNION
        SELECT model_id, zone_name FROM zone_min_z
      ) k
      LEFT JOIN zone_floor_area zfa
        ON k.model_id = zfa.model_id
        AND k.zone_name = zfa.zone_name
      LEFT JOIN zone_min_z zmz
        ON k.model_id = zmz.model_id
        AND k.zone_name = zmz.zone_name
    ),

    /* 3) Calculate ceiling_height from wall heights in zone ------------------ */
    zone_ceiling_height AS (
      SELECT
        b.model_id,
        b.zone_name,
        -- Calculate height as max(z) - min(z) for walls in this zone
        MAX(v.vertex_z_coordinate::double precision) - MIN(v.vertex_z_coordinate::double precision) AS ceiling_height
      FROM ep.buildingsurface_features b
      JOIN zone_base zb
        ON b.model_id = zb.model_id
       AND b.zone_name = zb.instance_name
      JOIN ep.ep_buildingsurface_detailed_vertices v ON v.parent_id = b.id
      WHERE b.zone_name IS NOT NULL
        AND LOWER(b.surface_type) = 'wall'
      GROUP BY b.model_id, b.zone_name
    ),

    /* 4) Calculate floor level (ground = 0) ----------------------------------- */
    zone_floor_level AS (
      SELECT
        model_id,
        zone_name,
        min_z_coordinate,
        -- Calculate floor level: round((min_z - ground_z) / typical_floor_height)
        ROUND(
          (min_z_coordinate - MIN(min_z_coordinate) OVER (PARTITION BY model_id)) /
          NULLIF(3.0, 0)  -- Assume 3m floor height as typical
        )::int AS floor_level
      FROM zone_geometry
      WHERE min_z_coordinate IS NOT NULL
    ),
    model_max_floor_level AS (
      SELECT model_id, MAX(floor_level) AS max_floor_level
      FROM zone_floor_level
      GROUP BY model_id
    ),

    zone_surface_geom AS (
      SELECT
        b.model_id,
        b.zone_name,
        b.instance_name AS surface_name,
        COALESCE(b.area, 0)::double precision AS area
      FROM ep.buildingsurface_features b
      JOIN zone_base zb
        ON b.model_id = zb.model_id
       AND b.zone_name = zb.instance_name
      WHERE b.zone_name IS NOT NULL
        AND b.outside_boundary_condition_code = 1
        AND LOWER(b.surface_type) = 'wall'
    ),
    zone_shading AS (
      SELECT
        g.model_id,
        g.zone_name,
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
      FROM zone_surface_geom g
      LEFT JOIN ep.surface_shading_metrics_view sm
        ON sm.model_id = g.model_id
       AND sm.buildingsurface_id = g.surface_name
    ),
    zone_shading_agg AS (
      SELECT
        model_id,
        zone_name,
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
      FROM zone_shading
      GROUP BY model_id, zone_name
    ),

    /* 5) Thermostat control and setpoints ------------------------------------- */
    zone_thermostat AS (
      SELECT
        t.model_id,
        t.zone_or_zonelist_name AS zone_name,
        t.control_1_name AS setpoint_name,
        t.control_type_schedule_name AS thermostat_schedule_name,
        1 AS has_thermostat
      FROM ep.ep_zonecontrol_thermostat t
    ),

    /* 6) HVAC equipment connections ------------------------------------------- */
    zone_hvac AS (
      SELECT
        h.model_id,
        h.zone_name,
        h.zone_conditioning_equipment_list_name AS equipment_list_name,
        h.zone_air_inlet_node_or_nodelist_name AS inlet_node_name,
        h.zone_air_exhaust_node_or_nodelist_name AS exhaust_node_name,
        1 AS has_hvac
      FROM ep.ep_zonehvac_equipmentconnections h
    ),

    /* 7) Water use equipment (peak flow rate) --------------------------------- */
    zone_wateruse AS (
      SELECT
        w.model_id,
        w.zone_name,
        SUM(w.peak_flow_rate) AS total_peak_flow_rate,
        COUNT(*) AS num_wateruse_objects
      FROM ep.ep_wateruse_equipment w
      GROUP BY w.model_id, w.zone_name
    ),

    /* 8) Zone infiltration (flow rate and schedules) -------------------------- */
    zone_infiltration_data AS (
      SELECT
        i.model_id,
        s.instance_name AS zone_name,
        AVG(i.flow_rate_per_exterior_surface_area) AS avg_infiltration_flow_rate,
        -- Get schedule name (text or one-hot encoded)
        %s AS infiltration_schedule_name,
        COUNT(*) AS num_infiltration_objects
      FROM ep.ep_zoneinfiltration_designflowrate i
      LEFT JOIN ep.ep_edge_spaceandspacelistnames s
        ON i.model_id = s.model_id
        AND i.zone_or_zonelist_or_space_or_spacelist_name = s.instance_name
      WHERE s.preferred_table = 'ep_space' OR s.instance_name IS NULL
      GROUP BY i.model_id, s.instance_name
    ),

    /* 9) People (occupancy) loads --------------------------------------------- */
    zone_people AS (
      SELECT
        p.model_id,
        s.instance_name AS zone_name,
        AVG(p.people_per_floor_area) AS avg_people_per_floor_area,
        COUNT(*) AS num_people_objects
      FROM ep.ep_people p
      LEFT JOIN ep.ep_edge_spaceandspacelistnames s
        ON p.model_id = s.model_id
        AND p.zone_or_zonelist_or_space_or_spacelist_name = s.instance_name
      WHERE s.preferred_table = 'ep_space'
      GROUP BY p.model_id, s.instance_name
    ),

    /* 10) Lighting loads ------------------------------------------------------ */
    zone_lights AS (
      SELECT
        l.model_id,
        s.instance_name AS zone_name,
        COUNT(*) AS num_lighting_objects
      FROM ep.ep_lights l
      LEFT JOIN ep.ep_edge_spaceandspacelistnames s
        ON l.model_id = s.model_id
        AND l.zone_or_zonelist_or_space_or_spacelist_name = s.instance_name
      WHERE s.preferred_table = 'ep_space'
      GROUP BY l.model_id, s.instance_name
    ),

    /* 11) Electric equipment loads -------------------------------------------- */
    zone_equipment AS (
      SELECT
        e.model_id,
        s.instance_name AS zone_name,
        COUNT(*) AS num_equipment_objects
      FROM ep.ep_electricequipment e
      LEFT JOIN ep.ep_edge_spaceandspacelistnames s
        ON e.model_id = s.model_id
        AND e.zone_or_zonelist_or_space_or_spacelist_name = s.instance_name
      WHERE s.preferred_table = 'ep_space'
      GROUP BY e.model_id, s.instance_name
    )

    /* 12) Main query - combine all zone information --------------------------- */
    SELECT
      zb.id,
      zb.model_id,
      zb.instance_name,

      -- Geometric properties
      ROUND(COALESCE(zg.floor_area, 0)::double precision * 1e2) / 1e2 AS floor_area,

      ROUND(COALESCE(zch.ceiling_height, 0)::double precision * 1e2) / 1e2 AS ceiling_height,

      -- Floor level (ground = 0)
      COALESCE(zfl.floor_level, 0)::int AS floor_level,
      CASE WHEN COALESCE(zfl.floor_level, 0) = 0 THEN 1 ELSE 0 END::int AS is_ground_floor,
      CASE
        WHEN COALESCE(zfl.floor_level, 0) = COALESCE(mfl.max_floor_level, 0) THEN 1
        ELSE 0
      END::int AS is_top_floor,

      -- Optional shading proxy aggregates
      COALESCE(zs.shade_count, 0)::int AS shade_count,
      COALESCE(zs.shade_ray_count, 0)::int AS shade_ray_count,
      ROUND(COALESCE(zs.shade_min_dist, 0)::double precision * 1e6) / 1e6 AS shade_min_dist,
      ROUND(COALESCE(zs.shade_mean_dist, 0)::double precision * 1e6) / 1e6 AS shade_mean_dist,
      ROUND(COALESCE(zs.shade_p90_dist, 0)::double precision * 1e6) / 1e6 AS shade_p90_dist,
      ROUND(COALESCE(zs.shade_ray_dir_mean_sin, 0)::double precision * 1e6) / 1e6 AS shade_ray_dir_mean_sin,
      ROUND(COALESCE(zs.shade_ray_dir_mean_cos, 0)::double precision * 1e6) / 1e6 AS shade_ray_dir_mean_cos,
      ROUND(COALESCE(zs.shade_ray_elevation_mean_rad, 0)::double precision * 1e6) / 1e6 AS shade_ray_elevation_mean_rad,
      ROUND(COALESCE(zs.shade_ray_elevation_mean_sin, 0)::double precision * 1e6) / 1e6 AS shade_ray_elevation_mean_sin,
      ROUND(COALESCE(zs.shade_ray_elevation_mean_cos, 0)::double precision * 1e6) / 1e6 AS shade_ray_elevation_mean_cos,

      -- Thermostat and control
      COALESCE(zt.has_thermostat, 0)::int AS has_thermostat,
      zt.setpoint_name,
      zt.thermostat_schedule_name,

      -- HVAC system
      COALESCE(zh.has_hvac, 0)::int AS has_hvac,
      zh.equipment_list_name,
      zh.inlet_node_name,
      zh.exhaust_node_name,

      -- Water use
      ROUND(COALESCE(zw.total_peak_flow_rate, 0)::double precision * 1e8) / 1e8 AS peak_flow_rate,

      COALESCE(zw.num_wateruse_objects, 0)::int AS num_wateruse_objects,

      -- Infiltration
      ROUND(COALESCE(zi.avg_infiltration_flow_rate, 0)::double precision * 1e8) / 1e8 AS infiltration_flow_rate,
      zi.infiltration_schedule_name,
      COALESCE(zi.num_infiltration_objects, 0)::int AS num_infiltration_objects,

      -- Occupancy and loads
      ROUND(COALESCE(zp.avg_people_per_floor_area, 0)::double precision * 1e6) / 1e6 AS people_per_floor_area,
      COALESCE(zp.num_people_objects, 0)::int AS num_people_objects,
      COALESCE(zl.num_lighting_objects, 0)::int AS num_lighting_objects,
      COALESCE(ze.num_equipment_objects, 0)::int AS num_equipment_objects,

      -- Boolean indicators
      CASE WHEN zt.has_thermostat = 1 THEN 1 ELSE 0 END::int AS is_conditioned,
      CASE WHEN zp.num_people_objects > 0 THEN 1 ELSE 0 END::int AS is_occupied,
      CASE WHEN zl.num_lighting_objects > 0 THEN 1 ELSE 0 END::int AS has_lighting,
      CASE WHEN ze.num_equipment_objects > 0 THEN 1 ELSE 0 END::int AS has_equipment

    FROM zone_base zb
    LEFT JOIN zone_geometry zg
      ON zb.model_id = zg.model_id
      AND zb.instance_name = zg.zone_name
    LEFT JOIN zone_ceiling_height zch
      ON zb.model_id = zch.model_id
      AND zb.instance_name = zch.zone_name
    LEFT JOIN zone_floor_level zfl
      ON zb.model_id = zfl.model_id
      AND zb.instance_name = zfl.zone_name
    LEFT JOIN model_max_floor_level mfl
      ON zb.model_id = mfl.model_id
    LEFT JOIN zone_shading_agg zs
      ON zb.model_id = zs.model_id
      AND zb.instance_name = zs.zone_name
    LEFT JOIN zone_thermostat zt
      ON zb.model_id = zt.model_id
      AND zb.instance_name = zt.zone_name
    LEFT JOIN zone_hvac zh
      ON zb.model_id = zh.model_id
      AND zb.instance_name = zh.zone_name
    LEFT JOIN zone_wateruse zw
      ON zb.model_id = zw.model_id
      AND zb.instance_name = zw.zone_name
    LEFT JOIN zone_infiltration_data zi
      ON zb.model_id = zi.model_id
      AND zb.instance_name = zi.zone_name
    LEFT JOIN zone_people zp
      ON zb.model_id = zp.model_id
      AND zb.instance_name = zp.zone_name
    LEFT JOIN zone_lights zl
      ON zb.model_id = zl.model_id
      AND zb.instance_name = zl.zone_name
    LEFT JOIN zone_equipment ze
      ON zb.model_id = ze.model_id
      AND zb.instance_name = ze.zone_name
  $SQL$, infiltration_schedule_sql);

  -- Cache the SELECT-body so we can rebuild ep.comp_zone without duplicating SQL
  SELECT pg_get_viewdef(to_regclass('ep.comp_zone'), true) INTO base_select;
  IF base_select IS NULL OR btrim(base_select) = '' THEN
    RAISE EXCEPTION 'Could not read view definition for ep.comp_zone';
  END IF;
  base_select := regexp_replace(btrim(base_select), E';\\s*$', '');

  -- Materialize once (small: one row per zone) so variance checks are fast
  EXECUTE 'DROP TABLE IF EXISTS comp_zone_tmp';
  EXECUTE format('CREATE TEMP TABLE comp_zone_tmp ON COMMIT DROP AS %s', base_select);

  -- 2) Build a pruned view (drop columns with no variance) -------------------
  FOR col IN
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='ep' AND table_name='comp_zone'
    ORDER BY ordinal_position
  LOOP
    IF col.column_name = ANY(keep_cols) THEN
      select_list := select_list || format('t.%I, ', col.column_name);
    ELSE
      EXECUTE format(
        'SELECT COUNT(DISTINCT %I) FROM comp_zone_tmp WHERE %I IS NOT NULL',
        col.column_name, col.column_name
      ) INTO distinct_count;
      IF distinct_count > 1 THEN
        select_list := select_list || format('t.%I, ', col.column_name);
      END IF;
    END IF;
  END LOOP;

  select_list := regexp_replace(select_list, E',\\s*$', '');

  IF select_list IS NULL OR btrim(select_list) = '' THEN
    select_list := 't.id, t.model_id, t.instance_name';
  END IF;

  -- Cache once (small: one row per zone) so graph builds don't re-run the
  -- expensive geometry + joins for every model fetch.
  EXECUTE 'DROP TABLE IF EXISTS ep.comp_zone_cache CASCADE';
  EXECUTE 'CREATE TABLE ep.comp_zone_cache AS SELECT * FROM comp_zone_tmp';
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_comp_zone_cache_model_id ON ep.comp_zone_cache (model_id)';
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_comp_zone_cache_model_instance ON ep.comp_zone_cache (model_id, instance_name)';

  -- Rebuild ep.comp_zone with pruned columns (thin view over cache)
  EXECUTE 'DROP VIEW IF EXISTS ep.comp_zone CASCADE';
  EXECUTE format('CREATE VIEW ep.comp_zone AS SELECT %s FROM ep.comp_zone_cache t', select_list);

  EXECUTE 'DROP TABLE IF EXISTS comp_zone_tmp';
END$$;
