# The database

The graph configurations are defined as SQL over an ingested EnergyPlus model set: each node and edge
table is the output of a transform in `beg/sql/transforms/`, and a configuration is a declared
selection of those tables. Changing a representation means changing SQL, not Python.

---

## Connecting with a SQL client

Browsing the database directly — DBeaver, pgAdmin, DataGrip, psql — is the quickest way to see what
ingestion produced. The settings all come from the `BEG_DSN` variable in your `.env`, and this prints
them in the form a GUI client asks for:

```bash
beg db url --parts
```

```
  Host      127.0.0.1
  Port      5432
  Database  beg
  Username  beg
  Password  see POSTGRES_PASSWORD in your .env file
  Schemas   ep, ep_transformed
  Driver    PostgreSQL
```

In DBeaver: **Database → New Database Connection → PostgreSQL**, fill in the fields above, then open
**Databases → beg → Schemas**. The `ep` schema holds the ingested models, `ep_transformed` the node
and edge tables.

For a picture of the schema, the last cell of `BEG_01_Ingest` writes
`outputs/ingest/ep_schema_preview.html`: the most connected tables as linked blocks, every relation
listed underneath, and DBML you can paste into [dbdiagram.io](https://dbdiagram.io). It is generated
from the live database by `beg.report.db_schema_preview`.

### The password

`.env` starts as a copy of `.env.example`, which sets `POSTGRES_PASSWORD=change-me`. When the
container first started, Docker read that variable and PostgreSQL initialised its data volume with
it. To see yours:

```bash
grep POSTGRES_PASSWORD .env
```

- **PostgreSQL applies it only when the data volume is created.** Editing `.env` afterwards leaves
  the running database with the password it was initialised with. To change it, either
  `ALTER USER beg PASSWORD '…'` inside the database, or `docker compose down -v` to delete the
  volume and start over, which also deletes every ingested model.
- **The container is reachable only from this machine.** It publishes on `127.0.0.1`, and the
  database holds public data that re-running the notebooks reproduces. Before publishing the port to
  other machines — changing the binding in `docker-compose.yml` — set a strong password on a freshly
  created volume.

Anything that prints or writes the connection goes through `beg.settings.redacted_dsn()`, which
masks the password.

### Changing the port

The bundled container uses **5432**, the standard PostgreSQL port. If another server already listens
there — another project's server, a system PostgreSQL, Postgres.app — choose a different port.
`POSTGRES_PORT` in `.env` and the port inside `BEG_DSN` must agree:

```
BEG_DSN=postgresql+psycopg2://beg:your-password@127.0.0.1:5433/beg
POSTGRES_PORT=5433
```

Then `docker compose up -d` recreates the container on the new port; the data volume is kept.

With two servers on one port, connections can reach the other server. The symptom is
`password authentication failed for user "beg"` against a container that is running. To see what
holds the port:

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN
```

---

## Two schemas

| Schema | Written by | Contents |
|---|---|---|
| `ep` | `BEG_01_Ingest` | one table per EnergyPlus object type present in the model set, plus `ep_model` (the model registry) and `weather_city` (per-city EPW and design-day statistics) |
| `ep_transformed` | `BEG_02_Graphs` | the node and edge tables the configurations select from: `comp_zone`, `comp_element`, `comp_aperture`, `comp_base_element`, `zones_zone`, `building`, and the `edge_*` relations |

`beg.db.RAW_SCHEMA` and `beg.db.TRANSFORMED_SCHEMA` hold the two names.

`TRANSFORM_MODE` in `BEG_02_Graphs` chooses how the transforms are executed:

- `"tables"` (default, writes into `ep_transformed`) — for the full dataset and anything feeding
  PyTorch;
- `"views"` (writes into `ep`) — for small previews of a handful of models.

---

## The raw schema is dataset-dependent

Ingestion does more than copy epJSON into tables. It also:

- drops columns that carry no variation across the ingested models,
- coerces two-valued columns to numeric booleans,
- one-hot encodes low-cardinality categorical columns.

So **which columns exist in `ep` depends on which model set was ingested**. A column such as
`surface_type` may be present as text, as a family of one-hot columns (`surface_type_wall`,
`surface_type_floor`, …), or both, depending on the data. The SQL transforms therefore probe
`information_schema.columns` before referencing a column and assemble parts of their projection
dynamically; there are roughly seventy such probes.

Two consequences for interpreting results:

1. **Feature width is a property of the ingested dataset, not only of the configuration.**
   `comp_zone` additionally prunes any column with a single distinct value. Ingest a different model
   set and the same configuration can produce a different number of node features.
2. **Comparisons are only valid within one ingestion.** To compare configurations, build them all
   from the same `ep` schema, as `BEG_02_Graphs` does in a single run.

---

## Freshness: transforms are cached by existence, not by content

With `REBUILD_TRANSFORMS = False` in `BEG_02_Graphs`, a transform is skipped if its output relation
already exists, whatever the `.sql` file now says.

**After editing any file in `beg/sql/transforms/`, re-run `BEG_02_Graphs` with
`REBUILD_TRANSFORMS = True`**, or the stale table is reused. The shading precondition is the one
automatic exception: when it has to populate `surface_shading_metrics` itself, it rebuilds every
transform.

---

## Sizing

The `ep` schema is the largest part of the database: one row per EnergyPlus object across the whole
model set. `ep_transformed` is smaller but is rebuilt in full on every rebuild, so leave headroom for
both. `beg db status` reports the current size.

The generated artefacts outside the database, for the full dataset:

| Artefact | Size |
|---|---|
| `outputs/pyg_graphs/` | 4 GB, one file per model and configuration |
| `outputs/pyg_datasets/` | 3.6 GB, ten packages |
| `outputs/GNNruns/` | 13 GB for the 816 runs, mostly model checkpoints |

Server settings that matter for the heavier transforms (`comp_zone`, `buildingsurface_features`,
`construction_material_totals`): `shared_buffers`, `work_mem`, `maintenance_work_mem`, `max_wal_size`.
The bundled `docker-compose.yml` sets workable defaults; raise them if the machine allows.

---

## Resetting

Re-ingesting from scratch is the normal way to recover from a bad ingestion. `BEG_01_Ingest` does it
with `RESET_SCHEMA = True`; from Python:

```python
from beg.db.schema import create_schemas, drop_schemas

drop_schemas(confirm=True)   # deletes ep and ep_transformed and everything in them
create_schemas()
```

This touches only the database. The epJSON and weather inputs are untouched, so the state is
rebuilt by re-running `BEG_01_Ingest` and `BEG_02_Graphs`.

To drop only the transform outputs and keep the ingested models, drop `ep_transformed` alone and
re-run `BEG_02_Graphs` with `REBUILD_TRANSFORMS = True`.

## Backing up

With the bundled container:

```bash
docker compose exec -T db pg_dump -U beg -d beg --format=custom > beg.dump
docker compose exec -T db pg_restore -U beg -d beg --clean --if-exists < beg.dump
```

For another server, run `pg_dump` and `pg_restore` against the `BEG_DSN` URL without `+psycopg2`
(`postgresql://beg@127.0.0.1:5432/beg`).

A dump moves an ingested database to another machine without re-running `BEG_01_Ingest`.
