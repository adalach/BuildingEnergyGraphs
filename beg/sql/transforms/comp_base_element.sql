-- ============================================================================
-- comp_base_element.sql (Unified Homogeneous Component View)
--   • Combines comp_element and comp_aperture into a single homogeneous node type
--   • All nodes have the same feature set for homogeneous GNN processing
--   • Uses element_type to distinguish between different building elements
--   • Preserves zone_name for same-zone edge creation
--   • Features that don't apply to certain types are set to 0 or NULL (see docs/limitations.md)
-- ============================================================================

DROP VIEW IF EXISTS ep.comp_base_element CASCADE;

CREATE VIEW ep.comp_base_element AS

/* 1) Elements (walls, floors, roofs, ceilings) ----------------------------- */
WITH elements AS (
  SELECT
    e.id,
    e.model_id,
    e.instance_name,
    e.construction_name,
    e.zone_name,

    -- Element type (text label for identification)
    CASE
      WHEN e.is_wall = 1 THEN 'wall'
      WHEN e.is_floor = 1 THEN 'floor'
      WHEN e.is_roof = 1 THEN 'roof'
      WHEN e.is_ceiling = 1 THEN 'ceiling'
      ELSE 'unknown'
    END AS element_type,

    -- Type indicators (combined for homogeneous representation)
    e.is_wall,
    e.is_floor,
    e.is_roof,
    e.is_ceiling,
    NULL::INTEGER AS is_window,
    NULL::INTEGER AS is_door,
    NULL::INTEGER AS is_glass_door,
    NULL::INTEGER AS is_tubular_daylight,
    CAST(0 AS INTEGER) AS is_aperture,  -- Not an aperture

    -- Boundary conditions (only for elements)
    e.is_outdoors,
    e.is_surface,
    e.is_zone AS is_zone_boundary,
    e.is_ground,
    e.is_adiabatic,

    -- Exposure (only for elements)
    e.sun_exposure,
    e.wind_exposure,

    -- Geometric properties
    e.area,

    -- Construction/material properties (standardized names from comp_element)
    e.total_thickness,
    e.r_value_layers,
    e.u_value_total,
    e.solar_transmittance,
    e.visible_transmittance,
    e.front_emissivity,
    e.back_emissivity,
    e.thermal_absorptance,
    e.solar_absorptance,
    e.visible_absorptance,
    e.density,
    e.specific_heat,
    e.outside_roughness_category,

    -- Optional shading proxy features
    e.shade_count,
    e.shade_ray_count,
    e.shade_min_dist,
    e.shade_mean_dist,
    e.shade_p90_dist,
    e.shade_ray_dir_mean_sin,
    e.shade_ray_dir_mean_cos,
    e.shade_ray_elevation_mean_rad,
    e.shade_ray_elevation_mean_sin,
    e.shade_ray_elevation_mean_cos

  FROM ep.comp_element e
),

/* 2) Apertures (windows, doors, etc.) -------------------------------------- */
apertures AS (
  SELECT
    a.id,
    a.model_id,
    a.instance_name,
    a.construction_name,

    -- Get zone_name from parent building surface
    -- This allows apertures to be linked to zones via their parent surface
    e.zone_name,

    -- Element type (text label for identification)
    CASE
      WHEN a.is_window = 1 THEN 'window'
      WHEN a.is_door = 1 THEN 'door'
      WHEN a.is_glass_door = 1 THEN 'glass_door'
      WHEN a.is_tubular_daylight = 1 THEN 'tubular_daylight'
      ELSE 'unknown_aperture'
    END AS element_type,

    -- Type indicators (combined for homogeneous representation)
    NULL::INTEGER AS is_wall,
    NULL::INTEGER AS is_floor,
    NULL::INTEGER AS is_roof,
    NULL::INTEGER AS is_ceiling,
    a.is_window,
    a.is_door,
    a.is_glass_door,
    a.is_tubular_daylight,
    CAST(1 AS INTEGER) AS is_aperture,  -- Is an aperture

    -- Boundary conditions (not applicable for apertures, set to NULL)
    NULL::INTEGER AS is_outdoors,
    NULL::INTEGER AS is_surface,
    NULL::INTEGER AS is_zone_boundary,
    NULL::INTEGER AS is_ground,
    NULL::INTEGER AS is_adiabatic,

    -- Exposure (not directly applicable for apertures, set to NULL)
    NULL::INTEGER AS sun_exposure,
    NULL::INTEGER AS wind_exposure,

    -- Geometric properties
    a.area,

    -- Construction/material properties (already using matched names)
    a.total_thickness,
    a.r_value_layers,
    a.u_value_total,
    a.solar_transmittance,
    a.visible_transmittance,
    a.front_emissivity,
    a.back_emissivity,
    a.thermal_absorptance,
    a.solar_absorptance,
    a.visible_absorptance,
    a.density,
    a.specific_heat,
    a.outside_roughness_category,

    -- Optional shading proxy features
    a.shade_count,
    a.shade_ray_count,
    a.shade_min_dist,
    a.shade_mean_dist,
    a.shade_p90_dist,
    a.shade_ray_dir_mean_sin,
    a.shade_ray_dir_mean_cos,
    a.shade_ray_elevation_mean_rad,
    a.shade_ray_elevation_mean_sin,
    a.shade_ray_elevation_mean_cos

  FROM ep.comp_aperture a
  -- Join to get zone_name from parent building surface
  LEFT JOIN ep.comp_element e
    ON a.model_id = e.model_id
    AND a.building_surface_name = e.instance_name
)

/* 3) Union elements and apertures ------------------------------------------ */
SELECT * FROM elements
UNION ALL
SELECT * FROM apertures
ORDER BY model_id, instance_name;
