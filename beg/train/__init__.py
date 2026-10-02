"""Training execution, the run registry, and run verification.

Requires the ``train`` extra. Import explicitly::

    from beg.train.runner import build_training_plan, run_training_plan
    from beg.train.dashboard import discover_run_inventory, write_run_index
    from beg.train.verified_runs import verify_runs

Runs are content-addressed: ``run_fingerprint`` combines a dataset fingerprint
with an implementation fingerprint over the model and training code and the
torch/CUDA build, so a plan skips work already done and never mixes results
across revisions of either.
"""
