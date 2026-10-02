# Building Energy Graphs

Reproducible pipeline for constructing **building energy graphs** from EnergyPlus models and for
measuring how graph representation design affects GNN-based building energy prediction.

A building energy model is turned into a graph in ten different ways — two levels of granularity
(zone, component), two graph types (heterogeneous, homogeneous) and three weather-attachment
variants — and the same set of graph neural networks is trained on each. The comparison is the
experiment: the representation, not the architecture, is the variable under study.

This repository accompanies the manuscript *"Building Energy Graphs: How Graph Representation
Design Impacts GNN-Based Building Energy Prediction"* (under review).

The notebooks are committed with their outputs: the full run on all 13,260 BEM4AI models, and the
816 training runs behind the published tables. Every step's tables and figures can be read on GitHub
without installing anything.

---

## The ten configurations

| ID | Level | Graph type | Weather | Node types |
|----|-------|-----------|---------|-----------|
| C1  | zone      | homogeneous   | none            | zone |
| C2  | zone      | homogeneous   | graph-level     | zone |
| C3  | zone      | heterogeneous | none            | zone, building |
| C4  | zone      | heterogeneous | in building node| zone, building |
| C5  | zone      | heterogeneous | separate node   | zone, building, weather |
| C6  | component | homogeneous   | none            | element |
| C7  | component | homogeneous   | graph-level     | element |
| C8  | component | heterogeneous | none            | zone, element, aperture, building |
| C9  | component | heterogeneous | in building node| zone, element, aperture, building |
| C10 | component | heterogeneous | separate node   | zone, element, aperture, building, weather |

C2 and C7 are derived from C1 and C6 by attaching a graph-level weather vector. The other eight are
declared in `configs/`: each file lists its SQL transforms, node tables and edge tables, directly or
by extending a file in `configs/base/`.

---

## Pipeline

```
epJSON models + EPW/DDY weather
        |  BEG_01_Ingest            beg.ingest
        v
PostgreSQL schema  ep.*                       one table per EnergyPlus object type
        |  BEG_02_Graphs            beg.sql, beg.pipeline, beg.build
        v
PostgreSQL schema  ep_transformed.*           node and edge tables per configuration
        |
        v
outputs/pyg_graphs/{homo,hetero}/<config>/    one PyG graph per model, and a manifest per graph type
        |  BEG_03_Datasets          beg.datasets
        v
outputs/pyg_datasets/<config>/training_ready.pt    fixed split, normalised targets
        |  BEG_04_Train             beg.train, beg.models
        v
outputs/GNNruns/<run>/                        one directory per training run
        |  BEG_05_Results           beg.report
        v
outputs/paper/{tables,figures}/               the manuscript's tables and figures
```

`BEG_02_Graphs` to `BEG_04_Train` reuse what is already on disk unless told to rebuild. A training
run is identified by a fingerprint over the dataset, the model and training code, the torch/CUDA
build and the hyperparameters. `BEG_04_Train` skips runs that are complete, including runs trained
under another torch/CUDA build with everything else equal.

---

## Getting started

**1. Get the code and create an environment.**

```bash
git clone https://github.com/adalach/BuildingEnergyGraphs.git
cd BuildingEnergyGraphs
python3 -m venv .venv
```

**2. Activate it.** Everything after this runs inside the virtual environment. Activation lasts for
one terminal session: in a new tab or window, activate again, or `beg`, `pip` and `jupyter` report
`command not found`.

```bash
source .venv/bin/activate
```

On Windows PowerShell that is `.venv\Scripts\Activate.ps1`; in `cmd`, `.venv\Scripts\activate.bat`.
The prompt then starts with `(.venv)`. Every command below also runs without activation from the
environment's own copy, for example `.venv/bin/beg doctor`.

**3. Install the package and start the database.**

```bash
pip install -e ".[all]"
cp .env.example .env
docker compose up -d
beg db init
beg doctor
```

`.env` holds the database connection every module reads, and its defaults match the bundled
container. To choose another password, change both `POSTGRES_PASSWORD` and the password inside
`BEG_DSN` before `docker compose up -d`: PostgreSQL applies it only when the database is first
created. If another PostgreSQL server already uses port 5432, set `POSTGRES_PORT=5433` and the same
port in `BEG_DSN`.

`beg doctor` reports anything missing, including which notebooks this environment can run and the
pip command for those it cannot. [`docs/installation.md`](docs/installation.md) covers the optional
extras and running PostgreSQL without Docker.

**4. Get the data.** The building models come from the BEM4AI dataset;
[`docs/data.md`](docs/data.md) says what to download and where it goes.

**5. Run the notebooks in order.** Start Jupyter from the activated environment, so the notebooks
use the interpreter the package was installed into:

```bash
jupyter lab
```

```
notebooks/BEG_01_Ingest.ipynb      epJSON + weather  ->  PostgreSQL
notebooks/BEG_02_Graphs.ipynb      SQL transforms    ->  graph DataFrames -> PyG graphs
notebooks/BEG_03_Datasets.ipynb    graphs + labels   ->  training-ready datasets and splits
notebooks/BEG_04_Train.ipynb       datasets          ->  trained runs
notebooks/BEG_05_Results.ipynb     runs              ->  comparison tables and figures
```

`beg db status` reports at any point what exists and which notebook produces the rest. Running a
notebook replaces its committed outputs with your own.

### A first run on a subset

The full experiment trains 816 runs, which took 92 hours on an RTX 3060. To check an installation
first, run the pipeline on 200 models. Give the subset its own output directory by adding
`BEG_OUTPUT_DIR=outputs/subset` to `.env`, and remove that line again before the full run. Then run
the notebooks with these settings, each in the notebook's first code cell unless noted:

| Notebook | Settings |
|---|---|
| `BEG_01_Ingest` | `MAX_MODELS = 200` |
| `BEG_04_Train` | `CONFIRM_TRAINING = True`; in the second code cell, `VARIANT_BUDGETS = dict.fromkeys(MODEL_KEYS, 1)` and `SMOKE_TEST = True` |
| `BEG_05_Results` | `EXPERIMENT_MAX_EPOCHS = 2` and `EXPECTED_VARIANTS = 1` |

This trains one variant of each model family per configuration, 34 runs of two epochs each, and the
whole subset takes a few minutes on a laptop. Two epochs exercise every stage but learn little:
expect poor scores, and a few diverged runs that `BEG_05_Results` lists as missing.
`BEG_01_Ingest` rebuilds the database on every run, so the full run starts from a clean database.

### Looking at the database

`beg db url --parts` prints the host, port, database, user and schemas to enter in DBeaver, pgAdmin
or DataGrip; the password is the `POSTGRES_PASSWORD` line in your `.env`.
[`docs/database.md`](docs/database.md) describes the two schemas. `BEG_01_Ingest` also writes
`outputs/ingest/ep_schema_preview.html`, a diagram of the schema as ingested.

### Re-running a notebook from the command line

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/BEG_01_Ingest.ipynb
```

---

## Layout

```
beg/            the package: config, db, ingest, sql, pipeline, build,
                datasets, models, train, report, validation
configs/        the eight declared graph configurations and their two base files
notebooks/      the five pipeline stages, with their outputs
scripts/        check_no_secrets.py         connection strings and home paths in tracked files
                pack_training_bundle.py     BEG_04_Train and its inputs, for another machine
                run_graph_quality_checks.py a quality report on the built graphs
paper_assets/   the runs behind the published tables, and the two figures outside the pipeline
tests/          contract and data-quality tests
docs/           installation, data, database, limitations, training on another machine
```

## Data

The building energy models come from the **BEM4AI** dataset
([10.14459/2026mp1856158](https://doi.org/10.14459/2026mp1856158), under review): 13,260 EnergyPlus
models across 713 European cities. BEM4AI provides download links for the EPW and DDY weather file
of each city; the files themselves come from Climate.OneBuilding. [`docs/data.md`](docs/data.md) says
where everything goes.

The database, the PyG graphs, the training-ready datasets and the training runs are not
distributed. They are reproducible from the inputs by running the notebooks, and
[`docs/database.md`](docs/database.md#sizing) states the disk space to expect.

## Requirements

Python 3.10+, PostgreSQL 14+ (16 recommended). Training needs PyTorch and PyTorch Geometric; the
data half of the pipeline installs and runs without them.

## Contributing

```bash
pre-commit install
```

The hooks run `ruff` and `scripts/check_no_secrets.py`, which rejects connection strings and
home-directory paths in tracked files, stored notebook outputs included.

## Citation

This repository accompanies the paper:

> **Building Energy Graphs: How Graph Representation Design Impacts GNN-Based Building Energy
> Prediction.** *Under review — the DOI and full reference will be added on acceptance.*

and consumes the [BEM4AI dataset](https://doi.org/10.14459/2026mp1856158). Machine-readable metadata
for both are in `CITATION.cff`; please cite the software and the paper together.

## License

The code — the `beg` package, the notebooks, configurations, SQL transforms, scripts and tests — is
released under the [MIT License](LICENSE).

The data files — `data/labels.csv`, and the files in `paper_assets/data/` apart from the ODYSSEE-MURE
values named below — are released under [CC BY 4.0](LICENSE-CC-BY-4.0.txt), the licence of the
BEM4AI dataset they derive from. Reuse requires attribution, for example:

> BuildingEnergyGraphs data, by Agata Dalach, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/).

Two files come from third parties and keep their own terms, set out in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): `data/Energy+.schema.epJSON` (EnergyPlus licence)
and `paper_assets/data/odyssee_household_heating_consumption_2023.csv` (ODYSSEE-MURE).
