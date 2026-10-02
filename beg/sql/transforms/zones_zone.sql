-- ============================================================================
-- zones_zone.sql (zone nodes of the zone-level configurations)
-- ============================================================================

DROP VIEW IF EXISTS ep.zones_zone CASCADE;

CREATE VIEW ep.zones_zone AS
WITH
/* 1) Base zones ------------------------------------------------------------ */
zone_base AS (
  SELECT
    z.id,
    z.model_id,
    z.instance_name AS zone_name,
    z.floor_area,
    z.ceiling_height,
    z.floor_level,
    z.is_ground_floor,
    z.is_top_floor
  FROM ep.comp_zone z
),

/* 2) Surface geometry ------------------------------------------------------ */
surface_geom AS (
  SELECT
    b.model_id,
    b.zone_name,
    b.instance_name AS surface_name,
    LOWER(b.surface_type) AS surface_type,
    b.outside_boundary_condition_code,
    COALESCE(b.area, 0)::double precision AS area
  FROM ep.buildingsurface_features b
  WHERE b.zone_name IS NOT NULL
),

/* 3) Wall / roof areas ----------------------------------------------------- */
wall_areas AS (
  SELECT
    model_id,
    zone_name,
    SUM(CASE WHEN surface_type = 'wall' THEN area ELSE 0 END) AS wall_area_total,
    SUM(CASE WHEN surface_type = 'wall' AND outside_boundary_condition_code = 1 THEN area ELSE 0 END) AS wall_area_ext,
    SUM(CASE WHEN surface_type = 'wall' AND outside_boundary_condition_code = 2 THEN area ELSE 0 END) AS wall_area_int
  FROM surface_geom
  GROUP BY model_id, zone_name
),
roof_areas AS (
  SELECT
    model_id,
    zone_name,
    SUM(CASE WHEN surface_type IN ('roof','ceiling','roofceiling') THEN area ELSE 0 END) AS roof_area_total
  FROM surface_geom
  GROUP BY model_id, zone_name
),

/* 4) Envelope boundary area ----------------------------------------------- */
zone_areas AS (
  SELECT
    zb.model_id,
    zb.zone_name,
    COALESCE(w.wall_area_total, 0) AS wall_area_total,
    COALESCE(w.wall_area_ext, 0)   AS wall_area_ext,
    COALESCE(w.wall_area_int, 0)   AS wall_area_int,
    COALESCE(r.roof_area_total, 0) AS roof_area_total,
    COALESCE(w.wall_area_total, 0)
      + COALESCE(zb.floor_area, 0)
      + COALESCE(r.roof_area_total, 0) AS room_boundary_area_total
  FROM zone_base zb
  LEFT JOIN wall_areas w ON zb.model_id = w.model_id AND zb.zone_name = w.zone_name
  LEFT JOIN roof_areas r ON zb.model_id = r.model_id AND zb.zone_name = r.zone_name
),

/* 5) Apertures ------------------------------------------------------------- */
aperture_r AS (
  SELECT
    a.*,
    CASE
      WHEN COALESCE(a.r_value_layers, 0) > 0 THEN a.r_value_layers
      WHEN COALESCE(a.u_value_total, 0) > 0 THEN 1.0 / a.u_value_total
      ELSE NULL
    END AS r_total
  FROM ep.comp_aperture a
),
window_areas AS (
  SELECT
    a.model_id,
    s.zone_name,
    SUM(CASE WHEN a.is_window = 1 THEN a.area ELSE 0 END) AS window_area_total,
    SUM(CASE WHEN a.is_window = 1 AND s.outside_boundary_condition_code = 1 THEN a.area ELSE 0 END) AS window_area_ext
  FROM aperture_r a
  JOIN surface_geom s
    ON s.model_id = a.model_id
   AND s.surface_name = a.building_surface_name
  WHERE s.surface_type = 'wall'
  GROUP BY a.model_id, s.zone_name
),

/* 6) Zone-level shading proxy ---------------------------------------------- */
zone_shading AS (
  SELECT
    s.model_id,
    s.zone_name,
    s.area,
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
  FROM surface_geom s
  LEFT JOIN ep.surface_shading_metrics_view sm
    ON sm.model_id = s.model_id
   AND sm.buildingsurface_id = s.surface_name
  WHERE
    s.outside_boundary_condition_code = 1
    AND s.surface_type = 'wall'
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

/* 7) Area-weighted R-values ------------------------------------------------ */
element_r AS (
  SELECT
    s.model_id,
    s.zone_name,
    s.surface_type,
    s.outside_boundary_condition_code,
    s.area,
    CASE
      WHEN COALESCE(e.r_value_layers, 0) > 0 THEN e.r_value_layers
      WHEN COALESCE(e.u_value_total, 0) > 0 THEN 1.0 / e.u_value_total
      ELSE NULL
    END AS r_total
  FROM surface_geom s
  LEFT JOIN ep.comp_element e
    ON e.model_id = s.model_id
   AND e.instance_name = s.surface_name
),
r_ext_wall AS (
  SELECT model_id, zone_name,
    SUM(area * r_total) / NULLIF(SUM(area), 0) AS r_ext_wall
  FROM element_r
  WHERE surface_type = 'wall'
    AND outside_boundary_condition_code = 1
    AND r_total IS NOT NULL
  GROUP BY model_id, zone_name
),
r_roof AS (
  SELECT model_id, zone_name,
    SUM(area * r_total) / NULLIF(SUM(area), 0) AS r_roof
  FROM element_r
  WHERE surface_type IN ('roof','ceiling','roofceiling')
    AND r_total IS NOT NULL
  GROUP BY model_id, zone_name
),
r_floor AS (
  SELECT model_id, zone_name,
    SUM(area * r_total) / NULLIF(SUM(area), 0) AS r_floor
  FROM element_r
  WHERE surface_type = 'floor'
    AND r_total IS NOT NULL
  GROUP BY model_id, zone_name
),
r_ext_window AS (
  SELECT
    a.model_id,
    s.zone_name,
    SUM(a.area * a.r_total) / NULLIF(SUM(a.area), 0) AS r_ext_window
  FROM aperture_r a
  JOIN surface_geom s
    ON s.model_id = a.model_id
   AND s.surface_name = a.building_surface_name
  WHERE a.is_window = 1
    AND s.outside_boundary_condition_code = 1
    AND a.r_total IS NOT NULL
  GROUP BY a.model_id, s.zone_name
)

/* 8) Final zone nodes ------------------------------------------------------ */
SELECT
  zb.id,
  zb.model_id,
  zb.zone_name AS instance_name,

  'zones'::text AS config_name,
  'zone'::text  AS node_type,
  (zb.model_id || '::' || zb.zone_name) AS node_uid,

  (ROUND(COALESCE(zb.floor_area,0) * 1e2) / 1e2) AS floor_area,
  (ROUND(COALESCE(zb.ceiling_height,0) * 1e2) / 1e2) AS ceiling_height,

  zb.floor_level,
  zb.is_ground_floor,
  zb.is_top_floor,

  (ROUND(za.wall_area_total * 1e4) / 1e4) AS wall_area_total,
  (ROUND(za.wall_area_ext   * 1e4) / 1e4) AS wall_area_ext,
  (ROUND(za.wall_area_int   * 1e4) / 1e4) AS wall_area_int,
  (ROUND(COALESCE(wa.window_area_ext,0) * 1e4) / 1e4) AS window_area_ext,
  (ROUND(za.roof_area_total * 1e4) / 1e4) AS roof_area_total,
  (ROUND(za.room_boundary_area_total * 1e4) / 1e4) AS room_boundary_area_total,

  COALESCE(zs.shade_count, 0)::int AS shade_count,
  COALESCE(zs.shade_ray_count, 0)::int AS shade_ray_count,
  (ROUND(COALESCE(zs.shade_min_dist, 0) * 1e6) / 1e6) AS shade_min_dist,
  (ROUND(COALESCE(zs.shade_mean_dist, 0) * 1e6) / 1e6) AS shade_mean_dist,
  (ROUND(COALESCE(zs.shade_p90_dist, 0) * 1e6) / 1e6) AS shade_p90_dist,
  (ROUND(COALESCE(zs.shade_ray_dir_mean_sin, 0) * 1e6) / 1e6) AS shade_ray_dir_mean_sin,
  (ROUND(COALESCE(zs.shade_ray_dir_mean_cos, 0) * 1e6) / 1e6) AS shade_ray_dir_mean_cos,
  (ROUND(COALESCE(zs.shade_ray_elevation_mean_rad, 0) * 1e6) / 1e6) AS shade_ray_elevation_mean_rad,
  (ROUND(COALESCE(zs.shade_ray_elevation_mean_sin, 0) * 1e6) / 1e6) AS shade_ray_elevation_mean_sin,
  (ROUND(COALESCE(zs.shade_ray_elevation_mean_cos, 0) * 1e6) / 1e6) AS shade_ray_elevation_mean_cos,

  (ROUND(
    COALESCE(wa.window_area_ext,0) / NULLIF(za.wall_area_ext,0) * 1e6
  ) / 1e6) AS wwr,

  (ROUND(
    za.wall_area_ext / NULLIF(za.wall_area_total,0) * 1e6
  ) / 1e6) AS ewr,

  (ROUND(
    za.wall_area_int / NULLIF(za.wall_area_total,0) * 1e6
  ) / 1e6) AS iwr,

  (ROUND(
    zb.floor_area / NULLIF(za.room_boundary_area_total,0) * 1e6
  ) / 1e6) AS frr,

  (ROUND(
    za.roof_area_total / NULLIF(za.room_boundary_area_total,0) * 1e6
  ) / 1e6) AS rrr,

  (ROUND(rew.r_ext_wall   * 1e6) / 1e6) AS r_ext_wall,
  (ROUND(rwin.r_ext_window* 1e6) / 1e6) AS r_ext_window,
  (ROUND(rrf.r_roof       * 1e6) / 1e6) AS r_roof,
  (ROUND(rfl.r_floor      * 1e6) / 1e6) AS r_floor

FROM zone_base zb
LEFT JOIN zone_areas   za   ON zb.model_id = za.model_id AND zb.zone_name = za.zone_name
LEFT JOIN window_areas wa   ON zb.model_id = wa.model_id AND zb.zone_name = wa.zone_name
LEFT JOIN zone_shading_agg zs ON zb.model_id = zs.model_id AND zb.zone_name = zs.zone_name
LEFT JOIN r_ext_wall   rew  ON zb.model_id = rew.model_id AND zb.zone_name = rew.zone_name
LEFT JOIN r_ext_window rwin ON zb.model_id = rwin.model_id AND zb.zone_name = rwin.zone_name
LEFT JOIN r_roof       rrf  ON zb.model_id = rrf.model_id AND zb.zone_name = rrf.zone_name
LEFT JOIN r_floor      rfl  ON zb.model_id = rfl.model_id AND zb.zone_name = rfl.zone_name;
