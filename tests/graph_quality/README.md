# Graph dataset quality checks

Tests that answer, for every generated graph dataset: *is data missing? do all node and
edge classes declared by the config exist? are there bad values, or features that could not
be calculated?* — across the whole chain

```
config JSON  →  graph DataFrames  →  NetworkX graphs  →  PyG graphs / training_ready.pt
```

The check logic lives in the reusable package **`beg/validation/`**; this folder holds
thin pytest wrappers, and `scripts/run_graph_quality_checks.py` is the command-line entry
point, which writes a Markdown + JSON report from the same checks.

## Quick start

```bash
# everything that is available on this machine (configs, PyG files, labels, ...)
pytest tests/graph_quality -q

# DataFrame-level checks straight from Postgres, 200 models per config
BEG_DSN=postgresql+psycopg2://... pytest tests/graph_quality --db --max-models 200 -q

# one config only, and treat warnings as failures
pytest tests/graph_quality --config component_hetero_external_weather --fail-on-warn -q

# stand-alone report (outputs/graph_quality/graph_quality_report.{md,json})
python scripts/run_graph_quality_checks.py --db --max-models 200
```

Every result is `PASS`, `WARN`, `FAIL` or `SKIP`. `FAIL` means the dataset must not be used as
is; `WARN` is suspicious but possibly legitimate; `SKIP` means the input for the check was not
available (no torch, no pickle, ...). Warnings surface as pytest warnings; `--fail-on-warn`
turns them into failures.

## Options

| option | meaning |
|---|---|
| `--config NAME` (repeatable) | restrict to config name(s), e.g. `zone_hetero_no_weather` |
| `--dataframes-pkl PATH` / `--with-dataframes` | run DataFrame checks on a pickled `build_graph_dataframes` result, if you have one (default `outputs/graph_dataframes.pkl`; loaded fully into RAM) |
| `--db` / `--db-schema` | fetch node/edge tables from Postgres instead (`BEG_DSN`), `--max-models` per config |
| `--nx-pickles PATH` | folder of NetworkX graphs pickled per model (one sub-folder per configuration), if you have them |
| `--pyg-graphs PATH`, `--pyg-datasets PATH` | PyG per-model graphs and `training_ready` packages |
| `--labels PATH` | labels csv (default `data/labels.csv`) |
| `--max-models N` | graphs loaded per config for NX/PyG checks (default 200) |

## What is checked

### Config (`test_configs.py`, `config_checks.py`) — no data needed
* strict schema (`validate_config`), lowercase prefix, file name `C<n>.json` names the same
  configuration as the prefix, 3-axis name `{zone|component}_{hetero|homo}_{weather}`,
  `type` ⇔ name, unique `id`s
* every node table and edge table has a producing SQL transform in `sql_transforms`
  (or is a known source table like `weather_city`); transform files exist and are registered
  in `PRIMARY_ORDER.txt`; `src_col`/`tgt_col` appear in the producing SQL
* edge endpoints are declared node tables, no duplicated relations, every node table is
  reachable through edges (connected *type graph*)
* weather variant ⇔ weather node tables / `weather_for` relations; homogeneous ⇒ one node
  table and self-type edges only

### DataFrames (`test_dataframes.py`, `dataframe_checks.py`)
* all config node/edge tables present and non-empty; key columns (`model_id`,
  `instance_name`, `src_col`, `tgt_col`) exist; non-numeric / bool / id-like columns flagged
* model coverage: `model_ids` = primary table, every table covers every model, singleton
  tables (`building`, `building_weather`, `weather_city`) have exactly one row per model
* node keys unique within a table **and across tables of the same model** (the NetworkX
  builder would silently merge equal `instance_name`s)
* edge integrity: dangling endpoints, self loops, duplicated rows, pairs stored in both
  directions although the config adds reverse edges, `belongs_to`/`part_of` children with
  0 or >1 parents, parents without children, nodes that appear in no edge table
* missing values: columns entirely NULL (feature could not be computed), partially NULL
  (the tensor builders `fillna(0)` silently), inf
* value domains for ~80 known columns (areas > 0, transmittances ∈ [0,1], U-values,
  monthly weather ranges, sin/cos, booleans 0/1, …) — see `spec.COLUMN_RULES`
* one-hot groups exclusive (`vintage_*`, `primary_function_*`, surface kind, boundary
  condition, aperture kind, unified relationship flags) and consistent with their label column
* degenerate features: constant columns, all-zero shading dataset, |x| > 1e7
* physics: wall area partition, `wwr`, ground/top floor flags vs `floor_level`, ground ⇒ no
  sun/wind exposure, windows transmit / opaque elements do not, U×R plausibility, apertures fit
  inside their parent element, zone `floor_area` ≈ summed floor elements, weather edges exactly
  cover exposed components (same rule as `edge_comp_env_weather.sql`)
* weather tables: `city_key` matches the model id, identical features per city, no all-zero rows

### NetworkX graphs (`test_nx_graphs.py`, `nx_checks.py`)
* metadata (`config_name`, `model_id` = file name), node types ⊆ config and all present,
  no `(missing_from_nodes)` sentinel nodes, `node_id` contiguous, singleton node types
* edge triples `(src_type, rel, tgt_type)` ⊆ expected (incl. reverse for `bidirectional`),
  every expected relation present, one-directional relations not reversed, forward/reverse
  symmetry, no self loops or duplicate parallel edges, hierarchical cardinality
* connectivity (weakly connected, isolated nodes, degree stats)
* attribute completeness per node type — the builder drops NaN attributes, so a node lacking
  a key that its siblings have is an uncomputable feature value; None/NaN/inf/bool/string
  values; domain rules; one-hot exclusivity; edge-attribute completeness and reverse-edge
  attribute equality
* semantics: weather edges exactly cover exposed components / windows, weather node connected,
  homogeneous `component_relationship` flags consistent with zone/aperture attributes
* derived graph-weather variants (C2/C7): `graph_feature_cols` / `_values` complete, numeric,
  in range, expected dimension (223)
* collection level: attribute schema identical across all graphs, constant attributes, node
  type coverage/statistics, unique model ids; **cross-config**: same building has the same zone
  count, component count, floor area, building attributes and weather statistics in every config

### PyG (`test_pyg_datasets.py`, `pyg_checks.py`) — needs torch + torch_geometric
* `Data`/`HeteroData`: x present, float, finite; `edge_index` shape/dtype/bounds, no self loops
  or duplicates; `edge_attr` rows = edges; reverse stores mirror forward stores; one-directional
  respected; isolated nodes; node/edge types match config; graph-level `weather_city` (dim 223,
  non-zero) for C2/C7
* collection: feature dimension stable per node type, all-zero / constant / huge columns
* `training_ready.pt`: lengths agree, ids unique, labels finite & positive, splits disjoint and
  covering, `*_ids` = `dataset_ids[*_idx]`, split sha recomputed with `stable_hash`, label
  mean/std recomputed from train ids, `graph.y` de-normalises to `labels_raw`, meta.json agrees;
  across configs: identical `dataset_ids`, splits and labels (paired comparison)

### Labels (`test_labels_and_cross.py`, `label_checks.py`)
* unique ids (also after case normalisation), finite, positive, plausible range, outliers,
  repeated values, every dataset model has a label (case-insensitive join flagged), unused labels

### Self-tests (`test_selfcheck_synthetic.py`, `test_selfcheck_pyg.py`) — always run
A synthetic C10-shaped dataset is built, passed through the *real* `build_nx_graph_for_model`,
and every check is exercised with a planted defect (dangling edge, NaN feature, non-exclusive
one-hot, missing weather edge, oversized aperture, …). These protect the checks themselves.

## Using the checks from a notebook

```python
from beg.validation import load_config_spec
from beg.validation.core import Report
from beg.validation.dataframe_checks import run_dataframe_checks
from beg.validation.nx_checks import run_graph_checks, check_graph_collection

spec = load_config_spec("configs/C10.json")
rep = Report(run_dataframe_checks(spec, out["data"]["component_hetero_external_weather"]))
rep.print_summary()                      # FAIL / WARN lines
rep.add(run_graph_checks(G, spec, expected_model_id=model_id))
print(rep.to_markdown())
```

## Adding a rule

* new feature column → add to `spec.COLUMN_RULES` (exact name) or `COLUMN_RULE_PATTERNS` (regex)
* new one-hot family → `spec.ONE_HOT_GROUPS`
* new node table that must be a singleton per model → `ConfigSpec.singleton_node_tables`
* new physical invariant → a block in `dataframe_checks.check_physical_consistency` (DataFrame
  level) and/or `nx_checks.check_semantics` (graph level), plus a planted-defect test in
  `test_selfcheck_synthetic.py`
