from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


BASE_TRAINING_CONFIG: dict[str, Any] = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
    "train_seed": 42,
}


CURRENT_MODEL_RUNS: list[dict[str, Any]] = [
    {"graph_kind": "homo", "model_key": "gcn", "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3}},
    {"graph_kind": "homo", "model_key": "sage_attn", "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3}},
    {"graph_kind": "homo", "model_key": "gin_mean", "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3}},
    {
        "graph_kind": "homo",
        "model_key": "gin_set2set",
        "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3, "set2set_steps": 3},
    },
    {
        "graph_kind": "hetero",
        "model_key": "rgcn",
        "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3, "pooling": "per_type_attention", "use_residual": True},
    },
    {
        "graph_kind": "hetero",
        "model_key": "heterosage",
        "model_overrides": {"hidden": 128, "dropout": 0.1, "num_layers": 3, "pooling": "per_type_attention"},
    },
]


MODEL_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "gcn": {
        "graph_kind": "homo",
        "description": "Homogeneous GCN baseline with mean pooling.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3, 4],
            "dropout": [0.1, 0.2, 0.3],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "sage_attn": {
        "graph_kind": "homo",
        "description": "Homogeneous GraphSAGE with attentional readout.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3, 4],
            "dropout": [0.1, 0.2, 0.3],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "gin_mean": {
        "graph_kind": "homo",
        "description": "Homogeneous GIN with mean pooling.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "gin_set2set": {
        "graph_kind": "homo",
        "description": "Homogeneous GIN with Set2Set readout.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "set2set_steps": [3, 5],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "rgcn": {
        "graph_kind": "hetero",
        "description": "Relation-specific linear message passing with optional residuals.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "pooling": ["per_type_attention", "per_type_mean"],
            "use_residual": [True, False],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "heterosage": {
        "graph_kind": "hetero",
        "description": "Heterogeneous GraphSAGE with per-type pooling.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "pooling": ["per_type_attention", "per_type_mean"],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3, 3e-3],
            "weight_decay": [1e-5, 1e-4],
        },
    },
    "heterogin": {
        "graph_kind": "hetero",
        "description": "Heterogeneous GIN using edge attributes through `GINEConv`.",
        "model_options": {
            "hidden": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.25],
            "pooling": ["per_type_attention", "per_type_mean"],
            "use_layernorm": [False, True],
        },
        "train_options": {
            "batch_size": [16, 32],
            "learning_rate": [3e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4],
            "grad_clip_norm": [0.0, 1.0],
        },
    },
}


def _stable_sample(items: list[dict[str, Any]], *, max_items: Optional[int], seed: int) -> list[dict[str, Any]]:
    if max_items is None or len(items) <= int(max_items):
        return items
    rng = __import__("random").Random(int(seed))
    chosen = sorted(rng.sample(range(len(items)), k=int(max_items)))
    return [items[i] for i in chosen]


def current_model_inventory_df() -> pd.DataFrame:
    rows = []
    for run in CURRENT_MODEL_RUNS:
        row = {
            "model_key": run["model_key"],
            "graph_kind": run["graph_kind"],
            **run["model_overrides"],
            **BASE_TRAINING_CONFIG,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["graph_kind", "model_key"]).reset_index(drop=True)


def proposed_search_space_df() -> pd.DataFrame:
    rows = []
    for model_key, spec in sorted(MODEL_SEARCH_SPACES.items()):
        model_options = {k: list(v) for k, v in spec["model_options"].items()}
        train_options = {k: list(v) for k, v in spec["train_options"].items()}
        rows.append(
            {
                "model_key": model_key,
                "graph_kind": spec["graph_kind"],
                "description": spec["description"],
                "model_options": model_options,
                "train_options": train_options,
            }
        )
    return pd.DataFrame(rows)


def expand_model_jobs(
    *,
    model_keys: Optional[Sequence[str]] = None,
    max_variants_per_model: Optional[int | Mapping[str, Optional[int]]] = None,
    seed: int = 42,
    train_overrides: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Expand a deterministic, bounded variant plan per model family.

    ``max_variants_per_model`` may be one cap for every selected family or a
    mapping such as ``{"gcn": 4, "rgcn": 8}``.  This keeps the notebook
    selection cell explicit about the compute budget for each model family.

    ``train_overrides`` replaces training hyperparameters on every job, which is
    how a notebook caps ``max_epochs`` for a short trial before committing to a
    long run. Because ``train_hparams`` is part of ``run_fingerprint``, a capped
    trial fingerprints differently from the real thing and cannot be mistaken
    for it — the real run is still queued afterwards.
    """
    selected = list(model_keys) if model_keys is not None else sorted(MODEL_SEARCH_SPACES)
    jobs: list[dict[str, Any]] = []

    for model_key in selected:
        if model_key not in MODEL_SEARCH_SPACES:
            raise KeyError(f"Unknown model_key='{model_key}'.")
        spec = MODEL_SEARCH_SPACES[model_key]

        model_keys_order = list(spec["model_options"].keys())
        train_keys_order = list(spec["train_options"].keys())
        model_products = list(product(*[spec["model_options"][k] for k in model_keys_order]))
        train_products = list(product(*[spec["train_options"][k] for k in train_keys_order]))

        variants = []
        for model_values in model_products:
            model_overrides = dict(zip(model_keys_order, model_values))
            for train_values in train_products:
                train_hparams = dict(BASE_TRAINING_CONFIG)
                train_hparams.update(dict(zip(train_keys_order, train_values)))
                train_hparams.update(dict(train_overrides or {}))
                variants.append(
                    {
                        "model_key": model_key,
                        "graph_kind": spec["graph_kind"],
                        "description": spec["description"],
                        "model_overrides": model_overrides,
                        "train_hparams": train_hparams,
                    }
                )

        if isinstance(max_variants_per_model, Mapping):
            max_items = max_variants_per_model.get(model_key)
        else:
            max_items = max_variants_per_model
        stable_key_seed = sum(ord(ch) for ch in str(model_key))
        variants = _stable_sample(variants, max_items=max_items, seed=seed + stable_key_seed)
        for variant_idx, variant in enumerate(variants, start=1):
            jobs.append({**variant, "variant_idx": int(variant_idx)})

    return jobs


def proposed_variant_counts_df() -> pd.DataFrame:
    rows = []
    for model_key, spec in sorted(MODEL_SEARCH_SPACES.items()):
        n_model = 1
        for values in spec["model_options"].values():
            n_model *= len(values)
        n_train = 1
        for values in spec["train_options"].values():
            n_train *= len(values)
        rows.append(
            {
                "model_key": model_key,
                "graph_kind": spec["graph_kind"],
                "n_model_variants": n_model,
                "n_train_variants": n_train,
                "n_total_variants": n_model * n_train,
            }
        )
    return pd.DataFrame(rows).sort_values(["graph_kind", "model_key"]).reset_index(drop=True)
