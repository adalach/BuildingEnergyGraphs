# Training on another machine

Training is the one stage of this pipeline that needs no database. `beg.train` and `beg.models` read
`outputs/pyg_datasets/` and nothing else: no PostgreSQL, no `.env`, no epJSON or EPW files, and not
the graphs in `outputs/pyg_graphs/`. Everything the models need is in the ten training-ready packages
`BEG_03_Datasets` writes.

So `BEG_04_Train`, the long stage, can run on whichever machine has the GPU while the data is
prepared on another.

## What moves

```
the repository (source only)          a few MB
outputs/pyg_datasets/<config>/        3.6 GB for the full set of ten
```

`scripts/pack_training_bundle.py` packs exactly that into one archive:

```bash
python scripts/pack_training_bundle.py
```

It writes `outputs/beg-train-windows.zip`, and stops without writing anything if it finds a
connection string or a path belonging to the machine that built it. `--no-datasets` packs the code
alone, to check the far machine's environment before moving gigabytes.

`docs/README-WINDOWS.md` is the step-by-step for a Windows target and is included in the archive. For
Linux or macOS the steps are the same: create a virtual environment, install PyTorch built for the
local accelerator, `pip install -e ".[train,notebooks]" matplotlib`, then run the notebook.

## What comes back

```
outputs/GNNruns/
```

One directory per run, plus `run_index.csv`:

```
run_meta.json          what was trained, on which hardware, with its fingerprints and metrics
history.csv            per-epoch train and validation loss
test_predictions.csv   per-model prediction and target
model_package.pt       the selected weights, with the model key and hyperparameters to rebuild it
model_state_dict.pt    the selected weights alone
last_checkpoint.pt     the state an interrupted run resumes from
_DONE.txt              written last, so a partial run is visible
```

About 13 GB for the full experiment, mostly checkpoints. Unpack it into `outputs/` on the machine
that runs `BEG_05_Results`. `BEG_04_Train` there counts these runs as done and trains nothing.

The packages are only read, so nothing else needs to come back.

## Runs from another torch/CUDA build

`run_fingerprint` combines a fingerprint of the dataset with one of the implementation. The
implementation fingerprint covers `torch.__version__` and `torch.version.cuda` as well as a hash of
`beg/models/gnn.py`, `beg/train/search.py` and `beg/train/runner.py`, so the same run trained under
two builds has two fingerprints and two run directories.

`match_runs_from_other_builds` connects them. For every planned run, it recomputes the fingerprint
under each torch/CUDA build that trained a run in `outputs/GNNruns/`, provided the source files
still hash as they did then. A completed run whose fingerprint matches counts as done: the plan marks
it `skip_done_other_build` and points at its directory.

Only completed runs are matched. A run left incomplete on one machine starts again from the
beginning on another.

Each `run_meta.json` also records a `hardware` block — device, device name, platform, torch and CUDA
versions — which `train_seconds` needs to be read correctly when runs come from more than one
machine.

## Determinism

The published runs were trained with `DETERMINISTIC = False`. They are reproducible in everything
the fingerprint covers — the data, the split, the code and the hyperparameters — but not bit-for-bit
in their weights. `DETERMINISTIC = True` selects deterministic kernels, which are slower, and
disables the CUDA autotuner.

## Speed

The graphs are small — 16 nodes on average at zone level, about 200 at component level — and are
batched 16 or 32 at a time. A GPU helps, but far less than it would on large models, because
assembling batches on the CPU is a real share of each epoch.
The data loader runs in the main process (`num_workers=0`), the setting that works on Windows.

The published experiment, 816 runs, took 92 hours of training on an RTX 3060, 74 of them on the six
heterogeneous configurations: an epoch of C10 took 21 seconds, one of C1 4 seconds. Run the smoke
test on the target machine before starting the full set.
