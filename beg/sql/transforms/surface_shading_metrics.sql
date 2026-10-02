-- ============================================================================
-- surface_shading_metrics.sql
--   • Exposes the shading proxy metrics that the published experiment requires
--   • Refuses to create a zero-valued fallback when the source table is absent
--   • Intended contract:
--       ep.surface_shading_metrics(
--         model_id,
--         buildingsurface_id,
--         zone_id,
--         shade_count,
--         ray_count,
--         min_dist,
--         mean_dist,
--         p90_dist,
--         ray_dir_mean_sin,
--         ray_dir_mean_cos,
--         ray_elevation_mean_rad,
--         ray_elevation_mean_sin,
--         ray_elevation_mean_cos
--       )
-- ============================================================================

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'ep'
      AND table_name = 'surface_shading_metrics'
  ) THEN
    DROP VIEW IF EXISTS ep.surface_shading_metrics_view CASCADE;
    CREATE VIEW ep.surface_shading_metrics_view AS
    SELECT
      model_id,
      buildingsurface_id,
      zone_id,
      shade_count,
      ray_count,
      min_dist,
      mean_dist,
      p90_dist,
      ray_dir_mean_sin,
      ray_dir_mean_cos,
      ray_elevation_mean_rad,
      ray_elevation_mean_sin,
      ray_elevation_mean_cos
    FROM ep.surface_shading_metrics;
  ELSE
    RAISE EXCEPTION USING
      MESSAGE = 'ep.surface_shading_metrics is required but missing',
      HINT = 'Run python -m beg.pipeline.shading before this transform; BEG_02_Graphs does this automatically.';
  END IF;
END$$;
