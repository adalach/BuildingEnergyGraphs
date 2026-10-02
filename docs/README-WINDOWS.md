# Training the BEG models on Windows with CUDA

This archive holds everything needed to run **`BEG_04_Train`** on a Windows machine with an NVIDIA
GPU: the package, the notebook and the ten training packages, which are all the training stage
reads. It needs no database, no EnergyPlus files and no `.env`.

When it is done, one folder goes back: `outputs\GNNruns\`.

---

## 0. What you need

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 |
| GPU | NVIDIA, with a current driver. Check with `nvidia-smi` |
| Disk | about 25 GB free: 3.6 GB of packages, the environment, and about 13 GB of results |
| Time | about 92 hours for the full experiment on an RTX 3060 |

Nothing here needs admin rights.

---

## 1. Unpack

Extract the archive somewhere short and without spaces, for example `C:\beg`. Then open
**PowerShell** in that folder — Shift + right-click the folder → *Open PowerShell window here*.

```powershell
dir
```

You should see `pyproject.toml`, `beg\`, `notebooks\` and `outputs\pyg_datasets\` **directly**
here. If you see a single folder called `beg-train-windows` instead, `cd beg-train-windows` first.

The archive contains a top-level `beg-train-windows\` folder, and Explorer's *Extract All* adds
another folder named after the zip, so extracting into a folder that is itself called
`beg-train-windows` gives `beg-train-windows\beg-train-windows\`. Work from whichever folder `dir`
shows `pyproject.toml` in.

---

## 2. Create the environment

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with *"running scripts is disabled on this system"*, allow it for this window
only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Your prompt should now start with `(.venv)`.

---

## 3. Install PyTorch **with CUDA**

`pip install torch` from PyPI installs the CPU-only build on Windows, so PyTorch comes from its own
index:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`cu124` suits most current drivers, and is the CUDA version the published runs were trained with. If
it fails, or `nvidia-smi` reports an older CUDA version, pick the matching command from
<https://pytorch.org/get-started/locally/>.

**Verify before going further:**

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

This must print `True`, a CUDA version, and your GPU's name — for example
`True 12.4 NVIDIA GeForce RTX 3060`. If it prints `False`, you have the CPU build — see
*Troubleshooting*.

A `UserWarning: Failed to initialize NumPy: No module named 'numpy'` is harmless at this point:
NumPy arrives in the next step.

---

## 4. Install the rest

```powershell
pip install -e ".[train,notebooks]" matplotlib
```

That is `torch-geometric`, `pandas`, `numpy`, `tqdm`, JupyterLab, and `matplotlib` for the coverage
figure. It keeps the torch installed in step 3.

Check that the package imports and can see its data:

```powershell
python -c "from beg.train.runner import load_training_ready_packages; from beg.settings import output_dir; print(len(load_training_ready_packages(output_dir() / 'pyg_datasets')), 'packages')"
```

Expected: `10 packages`.

---

## 5. Smoke test — do this before the long run

Open the notebook:

```powershell
jupyter lab notebooks\BEG_04_Train.ipynb
```

This occupies the PowerShell window: it starts a server, prints its URLs, and keeps running until you
press Ctrl+C twice in that window. Use a second PowerShell window if you need one.

In the **first code cell**, set:

```python
CONFIRM_TRAINING = True
```

In the **second code cell**, set:

```python
VARIANT_BUDGETS = dict.fromkeys(MODEL_KEYS, 1)
TARGET_CONFIGS = ["C1_zone_homo_no_weather", "C3_zone_hetero_no_weather"]
SMOKE_TEST = True
```

Run all four code cells. This trains seven runs for two epochs each — a few minutes — and proves the
whole path works: load, train, evaluate, write.

Check that the first cell's output says `device : CUDA (<your GPU>)`. If it says `CPU`, stop and fix
step 3.

The epoch cap is part of each run's fingerprint, so the smoke-test runs are kept apart from the real
ones, which are still planned afterwards.

To run the notebook without a browser:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\BEG_04_Train.ipynb
```

---

## 6. The real run

Back in the second code cell, restore:

```python
VARIANT_BUDGETS = dict.fromkeys(MODEL_KEYS, 24)
TARGET_CONFIGS = None
SMOKE_TEST = False
```

Run all four code cells again. This is **816 runs**: 24 variants of each of four model families on
the four homogeneous configurations, and of each of three families on the six heterogeneous ones.

**How long.** The published experiment took 92 hours on an RTX 3060, 74 of them on the six
heterogeneous configurations. Small graphs at batch sizes of 16 and 32 leave the GPU partly idle
while the CPU assembles batches.

**Stopping and restarting.** Each run writes `_DONE.txt` last, so an interrupted run is marked
incomplete. Re-running the notebook skips the finished runs; set `RETRY_INTERRUPTED = True` in the
second code cell to resume the interrupted ones from their last checkpoint. A run that fails is
recorded and the rest continue.

The third code cell shows the coverage matrix before anything runs — configurations against model
families, with the gaps visible. Read it before starting the run.

---

## 7. Send the results back

Only this folder matters:

```
outputs\GNNruns\
```

It holds one directory per run and `run_index.csv`, about 13 GB for the full experiment. Pack it with
the `tar` that ships with Windows:

```powershell
tar -cf GNNruns-results.tar -C outputs GNNruns
```

Copy `GNNruns-results.tar` back and unpack it into the main repository's `outputs` folder, where
`BEG_05_Results` reads it:

```bash
tar -xf GNNruns-results.tar -C outputs
```

The packages in `outputs\pyg_datasets\` are only read, so nothing else needs to come back.

---

## Troubleshooting

**`torch.cuda.is_available()` prints `False`.**
You have the CPU wheel. Uninstall and reinstall from the CUDA index:

```powershell
pip uninstall -y torch; pip install torch --index-url https://download.pytorch.org/whl/cu124
```

If it still prints `False`, run `nvidia-smi`. If that command is not found, the GPU driver is not
installed. If it reports a CUDA version below 12.4, use the matching `cu121` index instead.

**`Activate.ps1 cannot be loaded because running scripts is disabled`.**
See step 2 — `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` affects only this window.

**`does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found`.**
You are one folder above the project. Find it and change into its folder:

```powershell
Get-ChildItem -Recurse -Depth 2 -Filter pyproject.toml | Select-Object FullName
```

This is nearly always the double folder from extracting — see step 1. An activated virtual
environment keeps working from any folder, so just `cd` to the one that has `pyproject.toml`.

**`No training_ready.pt packages found`.**
PowerShell is not in the extracted folder, or the extraction was partial. `dir outputs\pyg_datasets`
should list ten directories named `C1_...` to `C10_...`, each containing `training_ready.pt`.

**`ModuleNotFoundError: No module named 'beg'` in the notebook.**
The kernel is a different Python from the one you installed into — usually a JupyterLab installed
outside this virtual environment. Run this in a cell:

```python
import sys; print(sys.executable)
```

It must be the `python.exe` inside your `.venv`. If it is not, stop the server (Ctrl+C twice), and
with the virtual environment active run `pip install jupyterlab`, then launch `jupyter lab` again.

**The execute cell stops with `This plan would train … runs`.**
`CONFIRM_TRAINING` is still `False` in the first code cell. Set it to `True` and run the cell again.

**It seems too slow, and `nvidia-smi` shows low GPU usage.**
Expected: with small graphs, the CPU-side batch assembly sets the pace, and the data loader runs in
the main process.

**A run fails.**
The failure and its message appear in the last cell's output and in that run's `run_meta.json`, and
the coverage matrix shows the gap. Send the results back anyway.
