# Paper assets

Where every figure and table in the manuscript comes from, and the data that the pipeline itself
does not produce.

Most figures are produced by the notebooks, under the file names the manuscript embeds, into
`outputs/paper/figures/` and `outputs/paper/tables/`. The two scripts in `figures/` produce the
figures that depend on something other than the pipeline's own outputs.

## Figures

| Manuscript file | Produced by | Input |
|---|---|---|
| `european_household_heating_consumption_2023.png` | `figures/household_heating_figure.py` | `paper_assets/data/odyssee_household_heating_consumption_2023.csv` |
| `footprint_area_and_total_zones_distributions.png` | `figures/dataset_creation_figures.py` | BEM4AI HBJSON models and `model_summary.csv` |
| `u_value_distributions_abcd_density.png` | `figures/dataset_creation_figures.py` | BEM4AI `model_summary.csv` |
| `eui_distribution_density.png` | `BEG_03_Datasets` | `data/labels.csv` |
| `graph_configs.png` | by hand; see below | one example building |
| `all_runs_validation_mse_grouped_by_weather.png` | `BEG_05_Results` | the runs in `outputs/GNNruns/` |
| `best_runs_training_curves_grid.png` | `BEG_05_Results` | the runs in `outputs/GNNruns/` |
| `best_runs_test_predictions_scatter.png` | `BEG_05_Results` | the runs in `outputs/GNNruns/` |
| `computational_costs.png` | `BEG_05_Results` | the runs and `paper_assets/data/bem4ai_simulation_timing.csv` |

`graph_configs.png` is assembled by hand for one example building. `BEG_02_Graphs` draws the
configuration topologies of any exported building into `outputs/figures/configuration_topologies.png`;
`FIGURE_MODEL` selects it.

Not generated here: `main-pipeline.png`, `bem-pipeline.png`, `beg-pipeline.png`,
`beg-configurations.png` and `shading-proxy.png` are diagrams, and `city_source_layers.png` and
`footprint-transformations.png` document the dataset's generation, which belongs to BEM4AI.

`BEG_05_Results` also writes two validation-MSE figures that show the spread of every run.

## Tables

Written by `BEG_05_Results` to `outputs/paper/tables/`:

| File | In the manuscript |
|---|---|
| `test_summary_best_runs.csv` | the selected-candidate table, C1\*–C10\* |
| `validation_summary_all_runs_config_model_key.csv` | the appendix summary over all stable runs |
| `parameter_count_ranges.csv` | the appendix parameter ranges per model family |
| `building_order_fidelity.csv` | the ranking of buildings by C5\* and C10\* |
| `computational_cost_summary.csv` | the cost comparison in the text |

The remaining tables — `configuration_ranking.csv`, `weather_variant_effect.csv`,
`best_runs_error_distribution.csv`, `training_cost.csv`, `prediction_time_upper_bound.csv` and
`parameter_counts_by_run.csv` — are the detail behind those numbers. `BEG_03_Datasets` writes
`eui_distribution_statistics.csv` beside them, the statistics of the EUI distribution, and
`figures/dataset_creation_figures.py` writes `dataset_creation_summary.json`: the footprint, zone and
perimeter statistics and the number of models per sector, building type and construction period.

## Data

| File | What it is |
|---|---|
| `published_runs.csv` | the 816 runs behind the tables, one row each, with metrics recomputed from the run's own output |
| `published_runs_provenance.json` | the fingerprint of the code that trained them; `tests/test_published_runs_provenance.py` fails if that code changes |
| `bem4ai_simulation_timing.csv` | EnergyPlus wall-clock time and SQL output size for each BEM4AI model, from the simulation campaign |
| `odyssee_household_heating_consumption_2023.csv` | 2023 climate-corrected household space-heating consumption per floor area for 24 European countries, as reported by ODYSSEE-MURE and cited in the figure caption |

## Running the scripts

```bash
python paper_assets/figures/household_heating_figure.py
```

```bash
python paper_assets/figures/dataset_creation_figures.py --workers 8
```

The second needs the BEM4AI HBJSON models unpacked into `data/hbjson/` (from `models/hbjson.zip`)
and the dataset's `summaries/model_summary.csv` copied into `data/summaries/`. Parsing all 13,260
models takes a while; `--statistics-input` reuses an earlier pass.
