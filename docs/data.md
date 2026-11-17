# Data

## What ships with the repository

| Path | Contents |
|---|---|
| `data/labels.csv` | the target values for all 13,260 buildings (`building_id,eui_kwh_per_m2`) |
| `data/Energy+.schema.epJSON` | the EnergyPlus input schema, redistributed under its own licence |
| `paper_assets/data/` | the runs behind the published tables, and the external data two figures need — see `paper_assets/README.md` |

## The BEM4AI dataset

The building energy models come from **BEM4AI**, a dataset of 13,260 EnergyPlus models across 713
European cities. The dataset is under review for publication and is available from the corresponding
author on request.

The pipeline needs two parts of it, placed into `data/`:

| From the BEM4AI archive | Into this repository |
|---|---|
| `models/epjson.zip` | unzipped into `data/`, giving `data/epjson/` |
| `weather/` | copied to `data/weather/` |

BEM4AI does not redistribute the weather files. `data/weather/README.md`, from the archive, lists
the Climate.OneBuilding downloads: put the ZIP files into `data/weather/downloads/` and run

```bash
cd data/weather
python3 extract_weather.py
```

which writes one EPW and one DDY file per city. The finished layout:

```
data/
  epjson/                 13,260 files, one per building, named <city>_<nnnn>.epjson
  weather/epw/            713 files, one .epw per city
  weather/ddy/            713 files, one .ddy per city
  labels.csv              ships with this repository
  Energy+.schema.epJSON   ships with this repository
```

Copy or move the files into place; a symlink records an absolute path that is meaningless on any
other machine. Everything under `data/` is ignored by git apart from the two files that ship with
the repository.

To keep the dataset elsewhere — it is about 7.4 GB unpacked — leave `data/` as it is and point the
pipeline at a directory with the same layout:

```bash
# in .env
BEG_DATA_DIR=/absolute/path/to/your/copy
```

Then re-run from `BEG_01_Ingest`.

### The EnergyPlus schema

`Energy+.schema.epJSON` is the input schema shipped with EnergyPlus, redistributed here under the
EnergyPlus licence — see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) — so that ingestion
works without a local EnergyPlus installation.

The schema drives table creation, so **its version matters**. A field the schema does not declare gets
no column and is dropped at ingestion without an error; a field the schema declares but the models do
not carry becomes an all-NULL column that the variance-pruning step then removes.

The copy shipped here is EnergyPlus **25.1** (849 object types), the version OpenStudio 3.10.0
bundles. The BEM4AI models declare version 24.2; the 25.1 schema declares every field they contain
(checked on 80 models, 394 distinct object–field pairs). Another schema version changes the raw `ep`
column inventory: the 23.2 schema, for example, lacks `Sizing:Zone.type_of_space_sum_to_use`,
`Lights.watts_per_floor_area` and `ElectricEquipment.watts_per_floor_area`.

## What is not distributed

The database, the PyG graphs, the training-ready datasets and the training runs are derived and
large. They are reproduced by running the notebooks; `docs/database.md` states the disk space to plan
for.

The one derived artefact kept in the repository is `paper_assets/data/published_runs.csv`, the runs
the published tables are computed from. Beside it, `published_runs_provenance.json` records the
fingerprint of the code that trained those runs, and `tests/test_published_runs_provenance.py` fails
if that code changes.

## Provenance

What the generated artefacts record:

- the graph manifests list every graph file by configuration and model;
- each training-ready package records its configuration signature, the split hashes and the label
  normalisation, from which the dataset fingerprint is computed;
- each run directory carries `run_meta.json` with the dataset fingerprint, an implementation
  fingerprint over the model and training code and the torch/CUDA build, the resulting run
  fingerprint, and the hardware it ran on.

A run whose data, code or hyperparameters differ gets a different fingerprint, so results from
different revisions of the pipeline do not mix.
