# Known limitations

Properties of the pipeline that produced the published results. Changing any of them changes those
results, so they are recorded here. Read this page before interpreting feature values.

---

## Data representation

**Null thermal resistance becomes zero.** `zones_zone.r_ext_window` is NULL for zones that have no
exterior window, and NULL is filled with `0.0` when the tensors are built. Zero resistance reads as an
infinitely conductive window, which is the opposite of the intended meaning. In NetworkX graphs the
attribute is dropped from those nodes instead, so the two graph backends disagree about the same
zones.

**Type-specific flags on the homogeneous component graph are filled with zero.** In
`comp_base_element`, walls carry an explicit `NULL::INTEGER` for aperture-only flags and apertures for
element-only flags (`is_window`, `is_wall`, `sun_exposure`, `is_ground`, …). These become `0` in the
tensors, so "not applicable" and "false" are indistinguishable in configurations C6 and C7.

**Weather design-day statistics are missing for some cities.** `summer_wind_speed_mean_ms`,
`winter_wind_speed_mean_ms` and the corresponding wind-direction columns are absent for part of the
city set. Where absent they become `0.0` in the graph-level weather vector of C2 and C7 and in the
weather nodes of C5 and C10, so those graphs carry a zero where others carry a measurement.

**Design-day humidity conditions of different types are averaged together.** `summer_humidity_value_mean`
aggregates design-day humidity regardless of the condition type EnergyPlus recorded (enthalpy versus
wet-bulb), producing values three orders of magnitude apart between cities. The column is not
comparable across cities and should not be read as a single physical quantity.

**The weather vector of C2 and C7 is standardised before the split.** `beg.build.weather_vector`
computes the z-score means and standard deviations over every model `BEG_02_Graphs` builds, before
`BEG_03_Datasets` draws the train/validation/test split, so test-set weather statistics enter the
training features of C2 and C7.

**String attributes are dropped silently by the graph builders.** `vintage`, `primary_function`,
`surface_type`, `element_type`, the weather `country`/`source`/`wmo` fields, `config_name` and
`node_uid` survive into NetworkX node attributes but are discarded without warning when tensors are
built. They are identifiers and metadata rather than features.

**A few buildings have zones that touch nothing.** In 12 of the 13,132 multi-zone models the zones
split into two groups that share no surface, so the zone graph is disconnected: Aarhus_0087,
Alphen_aan_den_Rijn_0010, Brighton_0008, Colmar_0014, Douai_0020, Eindhoven_0005, Katowice_0003,
Lelystad_0001, Meaux_0008, Minden_0022, Nijmegen_0015 and Terrassa_0004. In Aarhus_0087, for example,
the two zones on each floor stand about 0.3 m apart with all their walls exposed to outdoors, so only
the floor/ceiling pairs connect zones. The models are kept as generated; message passing in these
graphs cannot cross between the groups at zone level.

**Model identifiers differ in case between artefacts.** PyG dataset ids are lowercase
(`a_coruna_0001`) while `labels.csv` uses the original casing (`A_Coruna_0001`). Joins work only
after normalisation, which every loader in `beg` performs; external consumers must do the same.

**Foreign keys on dropped columns disappear silently.** When post-processing drops a column that
carries a foreign key, PostgreSQL drops the constraint with it and reports nothing, so the
constraint count after post-processing is lower than the count reported before it. The transforms
join on the columns directly, and no result depends on the constraints.

---

## Pipeline behaviour

**Feature width depends on the ingested dataset.** Columns without variation are pruned at
ingestion, and low-cardinality categoricals are one-hot encoded there, so the number of node features
a configuration yields is a property of the model set, not only of the configuration. Comparisons are
valid within one ingestion. See `docs/database.md`.

**Reference columns keep their values.** Columns the EnergyPlus schema declares as `object_list` —
names of other objects, such as `Construction.layer_5` — pass through the three post-processing
passes unchanged and are dropped only when entirely NULL.

**Transform caching is by existence, not content.** With `REBUILD_TRANSFORMS = False`, a transform
whose output table exists is skipped, whether or not its SQL has changed since. Rebuild after editing
SQL.

**Transform ordering is maintained by hand.** `beg/sql/transforms/PRIMARY_ORDER.txt` is a flat list,
not a dependency graph. A new transform must be placed after everything it reads from; nothing checks
this.

---

## Scope

**One target.** The published experiments predict annual energy use intensity at graph level. The
pipeline carries the machinery for node-level and time-series targets only as far as the data model;
no such experiment is included.

**One model set.** All results come from a single BEM4AI ingestion. The representation ranking is
established on that set and has not been replicated on an independent one.

**Weather is climatological, not synchronised.** Weather features summarise the EPW and design-day
conditions of a building's city. They are not time-aligned with the simulation output, so no
configuration here supports temporal inference.
