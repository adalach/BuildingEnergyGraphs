# Installation

Two things must be in place: a Python environment, and a PostgreSQL server the pipeline can reach.

---

## 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

`.[all]` installs everything. The extras can also be installed separately, for a machine where
PyTorch is awkward to install or where only one stage runs:

| Extra | Gives you |
|---|---|
| *(none)* | ingestion, SQL transforms (including the shapely-based shading metrics), graph DataFrames, NetworkX graphs |
| `train` | PyTorch + PyTorch Geometric: PyG graph export, dataset packaging and GNN training |
| `viz` | matplotlib for every figure; plotly and Honeybee/Ladybug for the 3D building previews in the graph explorer |
| `notebooks` | IPython, the Jupyter kernel, JupyterLab and nbconvert |
| `dev` | pytest, ruff, pre-commit |

What each notebook needs, if you install less than `.[all]`:

| Notebook | Install |
|---|---|
| `BEG_01_Ingest` | `pip install -e ".[notebooks]" matplotlib` |
| `BEG_02_Graphs` | `pip install -e ".[notebooks,train,viz]"` |
| `BEG_03_Datasets` | `pip install -e ".[notebooks,train]" matplotlib` |
| `BEG_04_Train` | `pip install -e ".[notebooks,train]" matplotlib` |
| `BEG_05_Results` | `pip install -e ".[notebooks,train]" matplotlib` |

The first cell of every notebook checks its own requirements and, if something is missing, prints
the packages and the command that adds them. `beg doctor` prints the same check for all five
notebooks at once. After installing anything, restart the notebook kernel: a running kernel does not
see packages installed after it started.

To commit to this repository, install the hooks as well:

```bash
pre-commit install
```

---

## 2. Configure the connection

```bash
cp .env.example .env
```

`.env` is ignored by git. `BEG_DSN` in it is the only place the connection string exists; every
module, notebook and script reads it from there.

```
BEG_DSN=postgresql+psycopg2://beg:your-password@127.0.0.1:5432/beg
```

---

## 3. PostgreSQL

Pick whichever route suits the machine. All five end at the same place.

### a. Docker (recommended)

Nothing is installed on the host, the version is pinned, and removing it is one command.

```bash
docker compose up -d        # start (reads POSTGRES_* from .env)
docker compose logs -f db   # watch the server come up
docker compose down         # stop, keeping the data volume
docker compose down -v      # stop and delete the data volume
```

The bundled service is PostgreSQL 16 with `shared_buffers=1GB`, `work_mem=64MB`,
`maintenance_work_mem=512MB` and `max_wal_size=4GB`, which the larger transforms want. Raise them in
`docker-compose.yml` if you have the memory.

`POSTGRES_PORT` in `.env` decides which host port the container publishes, 5432 by default. If
another PostgreSQL server already listens there, choose another port, such as 5433, and use the same
port in `BEG_DSN`; see *Troubleshooting* below.

### b. macOS — Homebrew

```bash
brew install postgresql@16
brew services start postgresql@16          # starts now, and again at login
createuser -s beg && createdb -O beg beg
psql -d beg -c "ALTER USER beg WITH PASSWORD 'your-password';"
```

### c. macOS — Postgres.app

Install from <https://postgresapp.com>, open it once, and initialise the server. Postgres.app
listens only while it is open; enable *Start automatically* in its settings to have the server start
at login. Then create the role and the database as in *b*.

### d. Linux — system package

```bash
sudo apt install postgresql-16
sudo systemctl enable --now postgresql
sudo -u postgres createuser -s beg
sudo -u postgres createdb -O beg beg
sudo -u postgres psql -c "ALTER USER beg WITH PASSWORD 'your-password';"
```

### e. An existing server

Set `BEG_DSN` and nothing else. The account needs `CREATEDB` for `beg db init` to create the
database; if the database already exists, ordinary `CREATE SCHEMA` rights are enough.

---

## 4. Initialise and verify

```bash
beg db init      # creates the database (if needed) and the ep / ep_transformed schemas
beg doctor       # python version, installed extras, connection, directories
beg db status    # what exists in the database and on disk, and what produces the rest
```

`beg db init` is idempotent and never drops anything. Expected output on a fresh server:

```
database beg: created
schemas created: ep, ep_transformed
```

---

## Troubleshooting

**`BEG_DSN is not set`** — `.env` is missing or the variable is commented out. `beg db url` prints
what the project currently reads, with the password redacted.

**`connection refused`** — no server is listening. Docker: `docker compose ps`. Postgres.app: the
app is closed. Homebrew: `brew services list`. Linux: `systemctl status postgresql`.

**`password authentication failed for user "beg"` while `docker compose ps` says the container is
healthy, or `docker compose up` reports that the port is already in use** — another PostgreSQL on
this machine (Postgres.app, Homebrew, a system service) holds port 5432, and connections reach that
server, which has no `beg` role. Check who is listening:

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN     # macOS / Linux
```

Two entries means two servers. Give this project its own port:

```bash
# .env
POSTGRES_PORT=5433
BEG_DSN=postgresql+psycopg2://beg:your-password@127.0.0.1:5433/beg
```

```bash
docker compose up -d && beg db init
```

**`password authentication failed`, with only one server running** — the password in `.env` and the
one the server holds for that role disagree. With Docker, `POSTGRES_PASSWORD` takes effect only when
the data volume is first created, so editing `.env` afterwards changes nothing. Either
`docker compose down -v` (which deletes the database) or `ALTER USER` inside the container.

**`permission denied to create database`** — the role lacks `CREATEDB`. Either grant it, or create
the database by hand and re-run `beg db init`, which then only adds the schemas.

**Ingestion is slow or the server restarts under load** — raise `shared_buffers`, `work_mem` and
`maintenance_work_mem`; see `docs/database.md` for the disk and memory the full dataset needs.
