-- ============================================================================
-- Construction rollup with provenance (fully dynamic column handling)
-- Creates: ep.construction_material_totals
-- Notes:
--   • Dynamically checks for column existence (post-processing may remove columns)
--   • Handles missing material property columns gracefully
--   • Computes in double precision and rounds as ROUND(x * 1eN) / 1eN, since
--     ROUND(x, n) is not defined for double precision
-- ============================================================================

DO $$
DECLARE
  layer_cols text[];
  unpivot_sql text;

  -- Material table column flags
  mat_thickness boolean; mat_conductivity boolean; mat_thermal_abs boolean;
  mat_solar_abs boolean; mat_visible_abs boolean; mat_density boolean;
  mat_specific_heat boolean; mat_roughness boolean;

  nomass_thermal_r boolean; nomass_thermal_abs boolean;
  nomass_solar_abs boolean; nomass_visible_abs boolean; nomass_roughness boolean;

  glz_thickness boolean; glz_conductivity boolean;
  glz_solar_trans boolean; glz_visible_trans boolean; glz_ir_trans boolean;
  glz_front_emiss boolean; glz_back_emiss boolean;

  gas_thickness boolean; gas_type boolean;

  sgs_shgc boolean; sgs_vis_trans boolean; sgs_u boolean;

  material_union_sql text;
  view_sql text;
BEGIN
  -- Check layer columns
  SELECT array_agg(column_name ORDER BY ordinal_position)
  INTO layer_cols
  FROM information_schema.columns
  WHERE table_schema = 'ep'
    AND table_name = 'ep_construction'
    AND column_name ~ '^(outside_layer|layer_[0-9]+)$';

  -- Build unpivoted_layers
  SELECT string_agg(
    CASE
      WHEN col = 'outside_layer' THEN
        'SELECT id AS construction_id, model_id, outside_layer AS layer, 1 AS idx FROM ep.ep_construction WHERE outside_layer IS NOT NULL AND outside_layer <> ''''
'
      ELSE
        format('SELECT id, model_id, %I, %s FROM ep.ep_construction WHERE %I IS NOT NULL AND %I <> ''''',
          col, regexp_replace(col, 'layer_', '')::text, col, col)
    END,
    E'\n  UNION ALL\n'
    ORDER BY CASE WHEN col = 'outside_layer' THEN 1 ELSE regexp_replace(col, 'layer_', '')::int + 1 END
  )
  INTO unpivot_sql
  FROM unnest(layer_cols) AS col;

  -- Check ep_material columns
  SELECT
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='thickness'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='conductivity'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='thermal_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='solar_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='visible_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='density'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='specific_heat'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material' AND column_name='roughness')
  INTO mat_thickness, mat_conductivity, mat_thermal_abs, mat_solar_abs, mat_visible_abs, mat_density, mat_specific_heat, mat_roughness;

  -- Check ep_material_nomass columns
  SELECT
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material_nomass' AND column_name='thermal_resistance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material_nomass' AND column_name='thermal_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material_nomass' AND column_name='solar_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material_nomass' AND column_name='visible_absorptance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_material_nomass' AND column_name='roughness')
  INTO nomass_thermal_r, nomass_thermal_abs, nomass_solar_abs, nomass_visible_abs, nomass_roughness;

  -- Check ep_windowmaterial_glazing columns
  SELECT
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='thickness'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='conductivity'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='solar_transmittance_at_normal_incidence'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='visible_transmittance_at_normal_incidence'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='infrared_transmittance_at_normal_incidence'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='front_side_infrared_hemispherical_emissivity'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_glazing' AND column_name='back_side_infrared_hemispherical_emissivity')
  INTO glz_thickness, glz_conductivity, glz_solar_trans, glz_visible_trans, glz_ir_trans, glz_front_emiss, glz_back_emiss;

  -- Check ep_windowmaterial_gas columns
  SELECT
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_gas' AND column_name='thickness'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_gas' AND column_name='gas_type')
  INTO gas_thickness, gas_type;

  -- Check ep_windowmaterial_simpleglazingsystem columns
  SELECT
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_simpleglazingsystem' AND column_name='solar_heat_gain_coefficient'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_simpleglazingsystem' AND column_name='visible_transmittance'),
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='ep' AND table_name='ep_windowmaterial_simpleglazingsystem' AND column_name='u_factor')
  INTO sgs_shgc, sgs_vis_trans, sgs_u;

  -- Build material_union SQL (double precision everywhere)
  material_union_sql := format($MAT$
  SELECT 'opaque'::text AS mtype, m.model_id, m.instance_name,
         %s AS thickness, %s AS conductivity, NULL::double precision AS thermal_resistance,
         %s AS thermal_absorptance, %s AS solar_absorptance, %s AS visible_absorptance,
         %s AS density, %s AS specific_heat, %s AS roughness,
         NULL::double precision AS solar_trans, NULL::double precision AS visible_trans, NULL::double precision AS ir_trans,
         NULL::double precision AS front_emiss, NULL::double precision AS back_emiss,
         NULL::double precision AS u_factor, NULL::text AS gas_type
  FROM ep.ep_material m
  UNION ALL
  SELECT 'nomass', n.model_id, n.instance_name,
         NULL::double precision, NULL::double precision, %s AS thermal_resistance,
         %s AS thermal_absorptance, %s AS solar_absorptance, %s AS visible_absorptance,
         NULL::double precision, NULL::double precision, %s AS roughness,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::double precision, NULL::text
  FROM ep.ep_material_nomass n
  UNION ALL
  SELECT 'glazing', g.model_id, g.instance_name,
         %s AS thickness, %s AS conductivity, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::text,
         %s AS solar_trans, %s AS visible_trans, %s AS ir_trans,
         %s AS front_emiss, %s AS back_emiss,
         NULL::double precision, NULL::text
  FROM ep.ep_windowmaterial_glazing g
  UNION ALL
  SELECT 'gas', w.model_id, w.instance_name,
         %s AS thickness, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::text,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::double precision,
         COALESCE(%s,'')::text AS gas_type
  FROM ep.ep_windowmaterial_gas w
  UNION ALL
  SELECT 'simple', s.model_id, s.instance_name,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::double precision,
         NULL::double precision, NULL::double precision, NULL::text,
         %s AS solar_trans, %s AS visible_trans,
         NULL::double precision, NULL::double precision, NULL::double precision,
         %s AS u_factor, NULL::text
  FROM ep.ep_windowmaterial_simpleglazingsystem s
$MAT$,
    -- ep_material
    CASE WHEN mat_thickness     THEN 'm.thickness::double precision'           ELSE 'NULL::double precision' END,
    CASE WHEN mat_conductivity  THEN 'm.conductivity::double precision'        ELSE 'NULL::double precision' END,
    CASE WHEN mat_thermal_abs   THEN 'm.thermal_absorptance::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN mat_solar_abs     THEN 'm.solar_absorptance::double precision'   ELSE 'NULL::double precision' END,
    CASE WHEN mat_visible_abs   THEN 'm.visible_absorptance::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN mat_density       THEN 'm.density::double precision'             ELSE 'NULL::double precision' END,
    CASE WHEN mat_specific_heat THEN 'm.specific_heat::double precision'       ELSE 'NULL::double precision' END,
    CASE WHEN mat_roughness     THEN 'm.roughness'                             ELSE 'NULL::text' END,
    -- ep_material_nomass
    CASE WHEN nomass_thermal_r   THEN 'n.thermal_resistance::double precision'   ELSE 'NULL::double precision' END,
    CASE WHEN nomass_thermal_abs THEN 'n.thermal_absorptance::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN nomass_solar_abs   THEN 'n.solar_absorptance::double precision'   ELSE 'NULL::double precision' END,
    CASE WHEN nomass_visible_abs THEN 'n.visible_absorptance::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN nomass_roughness   THEN 'n.roughness'                             ELSE 'NULL::text' END,
    -- ep_windowmaterial_glazing
    CASE WHEN glz_thickness    THEN 'g.thickness::double precision'                                ELSE 'NULL::double precision' END,
    CASE WHEN glz_conductivity THEN 'g.conductivity::double precision'                             ELSE 'NULL::double precision' END,
    CASE WHEN glz_solar_trans  THEN 'g.solar_transmittance_at_normal_incidence::double precision'  ELSE 'NULL::double precision' END,
    CASE WHEN glz_visible_trans THEN 'g.visible_transmittance_at_normal_incidence::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN glz_ir_trans     THEN 'g.infrared_transmittance_at_normal_incidence::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN glz_front_emiss  THEN 'g.front_side_infrared_hemispherical_emissivity::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN glz_back_emiss   THEN 'g.back_side_infrared_hemispherical_emissivity::double precision'  ELSE 'NULL::double precision' END,
    -- ep_windowmaterial_gas
    CASE WHEN gas_thickness THEN 'w.thickness::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN gas_type      THEN 'w.gas_type'                     ELSE 'NULL::text' END,
    -- ep_windowmaterial_simpleglazingsystem
    CASE WHEN sgs_shgc THEN 's.solar_heat_gain_coefficient::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN sgs_vis_trans THEN 's.visible_transmittance::double precision' ELSE 'NULL::double precision' END,
    CASE WHEN sgs_u THEN 's.u_factor::double precision' ELSE 'NULL::double precision' END
  );

  -- Build final view SQL (params and all rounding in double precision)
  view_sql := format($VIEW$
DROP VIEW IF EXISTS ep.construction_material_totals CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ep.construction_material_totals_cache CASCADE;

CREATE MATERIALIZED VIEW ep.construction_material_totals_cache AS
WITH params AS (
  SELECT
    0.020::double precision AS def_t_nomass_default, 0.040::double precision AS def_t_nomass_door,
    0.006::double precision AS def_t_sgs, 0.0127::double precision AS def_t_gas,
    0.600::double precision AS op_def_k_default, 1.000::double precision AS glz_def_k_default,
    0.026::double precision AS def_k_air, 0.016::double precision AS def_k_argon,
    0.025::double precision AS def_k_gas_generic, 0.040::double precision AS def_k_nomass_virtual,
    0.170::double precision AS r_film_wall_total, 0.140::double precision AS r_film_roof_total,
    0.170::double precision AS r_film_floor_total, 0.300::double precision AS r_film_window_total,
    0.15::double precision AS delta_u_window_frame_add,
    0.00::double precision AS op_def_solar_trans, 0.00::double precision AS op_def_visible_trans,
    0.00::double precision AS op_def_ir_trans, 0.90::double precision AS op_def_front_emiss,
    0.90::double precision AS op_def_back_emiss, 0.90::double precision AS op_def_therm_abs,
    0.70::double precision AS op_def_solar_abs, 0.70::double precision AS op_def_visible_abs,
    1200::double precision AS op_def_density, 900::double precision AS op_def_cp,
    0.55::double precision AS glz_def_solar_trans, 0.70::double precision AS glz_def_visible_trans,
    0.00::double precision AS glz_def_ir_trans, 0.84::double precision AS glz_def_front_emiss,
    0.84::double precision AS glz_def_back_emiss, 0.00::double precision AS glz_def_therm_abs,
    0.07::double precision AS glz_def_solar_abs, 0.08::double precision AS glz_def_visible_abs,
    2500::double precision AS glz_constr_def_density, 750::double precision AS glz_constr_def_cp
),
names_rhint AS (
  SELECT c.id AS construction_id, c.model_id, c.instance_name,
    COALESCE((regexp_match(c.instance_name, '-R([0-9]+(\.[0-9]+)?)$'))[1]::double precision, NULL) AS r_ip_hint
  FROM ep.ep_construction c
),
names_rhint_si AS (
  SELECT construction_id, model_id,
    CASE WHEN r_ip_hint IS NOT NULL THEN r_ip_hint / 5.678 ELSE NULL END AS r_si_hint
  FROM names_rhint
),
orientation AS (
  SELECT c.id AS construction_id, c.model_id,
    CASE
      WHEN c.instance_name ILIKE '%%roof%%' OR c.instance_name ILIKE '%%ceiling%%' THEN 'roof'
      WHEN c.instance_name ILIKE '%%floor%%' OR c.instance_name ILIKE '%%slab%%'
        OR c.instance_name ILIKE '%%ground%%' OR c.instance_name ILIKE '%%underground%%' THEN 'floor'
      ELSE 'wall'
    END AS assembly_type
  FROM ep.ep_construction c
),
unpivoted_layers AS (
  %s
),
material_union AS (
%s
),
layers_enriched AS (
  SELECT
    l.construction_id, l.model_id AS mdl, l.idx, u.mtype, u.instance_name AS material_name,
    CASE WHEN u.thickness IS NOT NULL THEN 'data'
         WHEN u.mtype = 'nomass' AND u.thermal_resistance IS NOT NULL THEN 'calc'
         WHEN u.mtype IN ('simple','gas') THEN 'default' ELSE 'default' END AS src_thickness,
    COALESCE(u.thickness,
      CASE WHEN u.mtype = 'nomass' AND u.thermal_resistance IS NOT NULL
           THEN u.thermal_resistance * (SELECT def_k_nomass_virtual FROM params)
           WHEN u.mtype = 'simple' THEN (SELECT def_t_sgs FROM params)
           WHEN u.mtype = 'gas' THEN (SELECT def_t_gas FROM params) ELSE 0 END) AS thickness,
    CASE WHEN u.conductivity IS NOT NULL THEN 'data' ELSE 'default' END AS src_conductivity,
    COALESCE(u.conductivity,
      CASE WHEN u.mtype = 'gas' THEN
             CASE UPPER(COALESCE(u.gas_type,''))
               WHEN 'AIR' THEN (SELECT def_k_air FROM params)
               WHEN 'ARGON' THEN (SELECT def_k_argon FROM params)
               ELSE (SELECT def_k_gas_generic FROM params) END
           WHEN u.mtype = 'glazing' THEN (SELECT glz_def_k_default FROM params)
           WHEN u.mtype = 'opaque' THEN (SELECT op_def_k_default FROM params)
           ELSE NULL END) AS conductivity,
    CASE WHEN u.thermal_resistance IS NOT NULL THEN 'data'
         WHEN COALESCE(u.thickness,0) > 0 AND COALESCE(u.conductivity,0) > 0 THEN 'calc'
         ELSE 'calc' END AS src_r_layer,
    COALESCE(u.thermal_resistance, NULL) AS r_explicit,
    CASE WHEN u.solar_trans IS NOT NULL THEN 'data' ELSE 'default' END AS src_solar_trans,
    COALESCE(u.solar_trans,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_solar_trans FROM params)
                   WHEN 'simple' THEN (SELECT glz_def_solar_trans FROM params)
                   WHEN 'gas' THEN 0 ELSE (SELECT op_def_solar_trans FROM params) END) AS solar_trans,
    CASE WHEN u.visible_trans IS NOT NULL THEN 'data' ELSE 'default' END AS src_visible_trans,
    COALESCE(u.visible_trans,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_visible_trans FROM params)
                   WHEN 'simple' THEN (SELECT glz_def_visible_trans FROM params)
                   WHEN 'gas' THEN 0 ELSE (SELECT op_def_visible_trans FROM params) END) AS visible_trans,
    CASE WHEN u.ir_trans IS NOT NULL THEN 'data' ELSE 'default' END AS src_ir_trans,
    COALESCE(u.ir_trans,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_ir_trans FROM params)
                   WHEN 'simple' THEN (SELECT glz_def_ir_trans FROM params)
                   WHEN 'gas' THEN 0 ELSE (SELECT op_def_ir_trans FROM params) END) AS ir_trans,
    CASE WHEN u.front_emiss IS NOT NULL THEN 'data' ELSE 'default' END AS src_front_emiss,
    COALESCE(u.front_emiss,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_front_emiss FROM params)
                   WHEN 'simple' THEN (SELECT glz_def_front_emiss FROM params)
                   WHEN 'gas' THEN 0 ELSE (SELECT op_def_front_emiss FROM params) END) AS front_emiss,
    CASE WHEN u.back_emiss IS NOT NULL THEN 'data' ELSE 'default' END AS src_back_emiss,
    COALESCE(u.back_emiss,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_back_emiss FROM params)
                   WHEN 'simple' THEN (SELECT glz_def_back_emiss FROM params)
                   WHEN 'gas' THEN 0 ELSE (SELECT op_def_back_emiss FROM params) END) AS back_emiss,
    CASE WHEN u.thermal_absorptance IS NOT NULL THEN 'data' ELSE 'default' END AS src_thermal_absorptance,
    COALESCE(u.thermal_absorptance,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_therm_abs FROM params)
                   ELSE (SELECT op_def_therm_abs FROM params) END) AS thermal_absorptance,
    CASE WHEN u.solar_absorptance IS NOT NULL THEN 'data' ELSE 'default' END AS src_solar_absorptance,
    COALESCE(u.solar_absorptance,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_solar_abs FROM params)
                   ELSE (SELECT op_def_solar_abs FROM params) END) AS solar_absorptance,
    CASE WHEN u.visible_absorptance IS NOT NULL THEN 'data' ELSE 'default' END AS src_visible_absorptance,
    COALESCE(u.visible_absorptance,
      CASE u.mtype WHEN 'glazing' THEN (SELECT glz_def_visible_abs FROM params)
                   ELSE (SELECT op_def_visible_abs FROM params) END) AS visible_absorptance,
    CASE WHEN u.density IS NOT NULL THEN 'data' WHEN u.mtype = 'opaque' THEN 'default' ELSE 'default' END AS src_density,
    COALESCE(u.density, CASE WHEN u.mtype = 'opaque' THEN (SELECT op_def_density FROM params) ELSE NULL END) AS density,
    CASE WHEN u.specific_heat IS NOT NULL THEN 'data' WHEN u.mtype = 'opaque' THEN 'default' ELSE 'default' END AS src_specific_heat,
    COALESCE(u.specific_heat, CASE WHEN u.mtype = 'opaque' THEN (SELECT op_def_cp FROM params) ELSE NULL END) AS specific_heat,
    u.u_factor,
    CASE WHEN u.roughness = 'Smooth' THEN 0 WHEN u.roughness = 'MediumSmooth' THEN 1
         WHEN u.roughness = 'MediumRough' THEN 2 WHEN u.roughness = 'Rough' THEN 3
         WHEN u.roughness = 'VeryRough' THEN 4
         WHEN u.roughness IS NULL AND u.mtype IN ('glazing','simple','gas') THEN 0 ELSE 0 END AS roughness_code
  FROM unpivoted_layers l
  JOIN material_union u ON u.model_id = l.model_id AND u.instance_name = l.layer
),
layers_derived AS (
  SELECT construction_id, mdl AS model_id, idx, mtype, material_name, thickness,
    CASE WHEN r_explicit IS NOT NULL THEN r_explicit ELSE thickness / NULLIF(conductivity,0) END AS r_layer,
    solar_trans, visible_trans, ir_trans, front_emiss, back_emiss,
    thermal_absorptance, solar_absorptance, visible_absorptance,
    density, specific_heat, u_factor, roughness_code,
    src_thickness, src_conductivity, src_r_layer,
    src_solar_trans, src_visible_trans, src_ir_trans, src_front_emiss, src_back_emiss,
    src_thermal_absorptance, src_solar_absorptance, src_visible_absorptance,
    src_density, src_specific_heat
  FROM layers_enriched
),
agg AS (
  SELECT construction_id, model_id,
    (ROUND(SUM(thickness) * 1e4) / 1e4) AS total_thickness,
    (ROUND(SUM(COALESCE(r_layer,0)) * 1e4) / 1e4) AS r_layers,
    MIN(u_factor) FILTER (WHERE u_factor IS NOT NULL) AS u_window_min,
    BOOL_OR(mtype IN ('glazing','simple','gas')) AS has_window_layers,
    BOOL_OR(mtype IN ('glazing','simple')) AS has_glass_layers,
    (ROUND((SUM(thickness * solar_trans) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_solar_trans,
    (ROUND((SUM(thickness * visible_trans) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_visible_trans,
    (ROUND((SUM(thickness * front_emiss) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_front_emiss,
    (ROUND((SUM(thickness * back_emiss) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_back_emiss,
    (ROUND((SUM(thickness * thermal_absorptance) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_thermal_absorptance,
    (ROUND((SUM(thickness * solar_absorptance) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_solar_absorptance,
    (ROUND((SUM(thickness * visible_absorptance) / NULLIF(SUM(thickness),0)) * 1e3) / 1e3) AS avg_visible_absorptance,
    (ROUND((SUM(thickness * density) / NULLIF(SUM(thickness) FILTER (WHERE density IS NOT NULL),0)) * 1e2) / 1e2) AS avg_density_raw,
    (ROUND((SUM(thickness * specific_heat) / NULLIF(SUM(thickness) FILTER (WHERE specific_heat IS NOT NULL),0)) * 1e2) / 1e2) AS avg_cp_raw,
    SUM(thickness) FILTER (WHERE mtype IN ('opaque','nomass')) AS t_opaque
  FROM layers_derived
  GROUP BY construction_id, model_id
),
outside AS (
  SELECT construction_id, model_id, COALESCE(roughness_code,0) AS outside_roughness_category
  FROM layers_derived WHERE idx = 1
),
films AS (
  SELECT o.construction_id, o.model_id,
    CASE assembly_type
      WHEN 'roof' THEN (SELECT r_film_roof_total FROM params)
      WHEN 'floor' THEN (SELECT r_film_floor_total FROM params)
      ELSE (SELECT r_film_wall_total FROM params)
    END AS r_film_opaque_total,
    (SELECT r_film_window_total FROM params) AS r_film_window_total
  FROM orientation o
)
SELECT
  c.id, c.model_id, c.instance_name,
  a.total_thickness,
  (ROUND(GREATEST(a.r_layers, COALESCE(nh.r_si_hint, 0)) * 1e4) / 1e4) AS r_value_layers,
  (ROUND(COALESCE(
    a.u_window_min,
    CASE WHEN a.has_window_layers
         THEN (1 / NULLIF(GREATEST(a.r_layers, COALESCE(nh.r_si_hint,0)) + f.r_film_window_total, 0)) + (SELECT delta_u_window_frame_add FROM params)
         ELSE NULL END,
    1 / NULLIF(GREATEST(a.r_layers, COALESCE(nh.r_si_hint,0)) + f.r_film_opaque_total, 0)
  ) * 1e2) / 1e2) AS u_value_total,
  COALESCE(a.avg_solar_trans, 0) AS avg_solar_trans,
  COALESCE(a.avg_visible_trans, 0) AS avg_visible_trans,
  COALESCE(a.avg_front_emiss, 0) AS avg_front_emiss,
  COALESCE(a.avg_back_emiss, 0) AS avg_back_emiss,
  COALESCE(a.avg_thermal_absorptance, 0) AS avg_thermal_absorptance,
  COALESCE(a.avg_solar_absorptance, 0) AS avg_solar_absorptance,
  COALESCE(a.avg_visible_absorptance, 0) AS avg_visible_absorptance,
  COALESCE(a.avg_density_raw,
    CASE WHEN a.t_opaque > 0 THEN (SELECT p.op_def_density FROM params p)
         WHEN a.has_glass_layers = TRUE THEN (SELECT p.glz_constr_def_density FROM params p)
         ELSE 0 END) AS avg_density,
  COALESCE(a.avg_cp_raw,
    CASE WHEN a.t_opaque > 0 THEN (SELECT p.op_def_cp FROM params p)
         WHEN a.has_glass_layers = TRUE THEN (SELECT p.glz_constr_def_cp FROM params p)
         ELSE 0 END) AS avg_specific_heat,
  COALESCE(o.outside_roughness_category, 0) AS outside_roughness_category
FROM ep.ep_construction c
LEFT JOIN agg a ON a.construction_id = c.id AND a.model_id = c.model_id
LEFT JOIN outside o ON o.construction_id = c.id AND o.model_id = c.model_id
LEFT JOIN names_rhint_si nh ON nh.construction_id = c.id AND nh.model_id = c.model_id
LEFT JOIN films f ON f.construction_id = c.id AND f.model_id = c.model_id;
$VIEW$, unpivot_sql, material_union_sql);

  EXECUTE view_sql;
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_construction_material_totals_cache_model_id ON ep.construction_material_totals_cache (model_id)';
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_construction_material_totals_cache_model_instance ON ep.construction_material_totals_cache (model_id, instance_name)';
  EXECUTE 'CREATE VIEW ep.construction_material_totals AS SELECT * FROM ep.construction_material_totals_cache';
  RAISE NOTICE 'Created materialized view ep.construction_material_totals_cache and view ep.construction_material_totals';
END$$;
