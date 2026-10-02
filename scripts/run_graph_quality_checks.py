#!/usr/bin/env python
"""
Run the graph dataset quality checks and write a Markdown + JSON report.

Examples
--------
Static config checks + NetworkX preview pickles + PyG artefacts found in the repo::

    python scripts/run_graph_quality_checks.py

DataFrame checks straight from Postgres (200 models per config)::

    python scripts/run_graph_quality_checks.py --db --max-models 200      # uses BEG_DSN

DataFrame checks from the pickled DataFrame cache (large!) for two configs only::

    python scripts/run_graph_quality_checks.py --dataframes-pkl outputs/graph_dataframes.pkl \
        --config component_hetero_external_weather --config zone_hetero_no_weather

Exit code is 1 when any check FAILs (2 with ``--fail-on-warn`` and warnings).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from beg.validation import (  # noqa: E402
    config_checks,
    cross_config_checks,
    dataframe_checks,
    label_checks,
    nx_checks,
)
from beg.validation.core import CheckResult, Report  # noqa: E402
from beg.validation.loaders import (  # noqa: E402
    RepoPaths,
    fetch_payload_from_db,
    iter_pyg_graph_files,
    list_nx_pickle_configs,
    list_pyg_config_dirs,
    list_training_ready,
    load_graph_dataframes,
    load_meta_json,
    load_nx_pickles,
    make_engine_from_env,
    resolve_spec_for_output_name,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="project root (auto-detected)")
    ap.add_argument("--configs-dir", default=None)
    ap.add_argument("--config", action="append", default=None, help="restrict to config name(s)")
    ap.add_argument("--dataframes-pkl", default=None, help="pickled build_graph_dataframes result to check")
    ap.add_argument("--db", action="store_true", help="fetch DataFrames from the configured database")
    ap.add_argument("--db-schema", default="ep_transformed")
    ap.add_argument("--max-models", type=int, default=200, help="models per config for DB fetch / PyG / NX samples")
    ap.add_argument("--nx-pickles", default=None)
    ap.add_argument("--skip-nx", action="store_true")
    ap.add_argument("--pyg-graphs", default=None)
    ap.add_argument("--pyg-datasets", default=None)
    ap.add_argument("--skip-pyg", action="store_true")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out-dir", default=None, help="where to write graph_quality_report.{md,json} (default outputs/graph_quality)")
    ap.add_argument("--include-pass", action="store_true", help="list PASS rows in the markdown report")
    ap.add_argument("--fail-on-warn", action="store_true")
    args = ap.parse_args(argv)

    paths = RepoPaths(Path(args.root) if args.root else None)
    if args.configs_dir:
        paths.configs_dir = Path(args.configs_dir)
    specs = paths.specs(args.config)
    all_specs = paths.specs()
    report = Report()

    # ---- 1. configs ------------------------------------------------------------------
    _log(f"config checks: {len(specs)} configs in {paths.configs_dir}")
    report.add(config_checks.check_configs_dir(paths.configs_dir, transforms_dir=paths.transforms_dir, names=args.config))

    # ---- 2. dataframes ---------------------------------------------------------------
    payloads: Dict[str, dict] = {}
    if args.db:
        engine = make_engine_from_env()
        if engine is None:
            report.add(CheckResult.skip("df.source", "BEG_DSN not set", scope="dataframe"))
        else:
            shared_ids = None
            for spec in specs.values():
                _log(f"fetching {spec.full_name} from {args.db_schema} (max {args.max_models})")
                try:
                    p = fetch_payload_from_db(engine, spec, schema=args.db_schema, max_models=args.max_models, model_ids=shared_ids)
                except Exception as exc:  # noqa: BLE001
                    report.add(CheckResult.fail("df.source", f"DB fetch failed: {exc}", config=spec.name, scope="dataframe"))
                    continue
                shared_ids = shared_ids or p["model_ids"]
                payloads[spec.name] = p
    elif args.dataframes_pkl:
        pkl = Path(args.dataframes_pkl)
        if pkl.exists():
            _log(f"loading {pkl} (this can take a while)")
            out = load_graph_dataframes(pkl)
            for name, p in out.get("data", {}).items():
                if name in specs:
                    payloads[name] = dataframe_checks.flatten_payload(p)
        else:
            report.add(CheckResult.skip("df.source", f"{pkl} not found", scope="dataframe"))
    for name, p in payloads.items():
        _log(f"dataframe checks: {name}")
        report.add(dataframe_checks.run_dataframe_checks(specs[name], p))
    if len(payloads) > 1:
        report.add(cross_config_checks.check_payloads_consistent(payloads, specs))

    # ---- 3. NetworkX pickles ---------------------------------------------------------
    nx_root = Path(args.nx_pickles) if args.nx_pickles else paths.nx_pickles_root
    if not args.skip_nx and nx_root.is_dir():
        per_model: Dict[str, Dict[str, object]] = {}
        for d in list_nx_pickle_configs(nx_root):
            spec, derived = resolve_spec_for_output_name(d.name, all_specs)
            if spec is None or (args.config and spec.name not in args.config):
                continue
            graphs = load_nx_pickles(d, limit=args.max_models)
            _log(f"nx checks: {d.name} ({len(graphs)} graphs)")
            for mid, G in graphs.items():
                report.add(nx_checks.run_graph_checks(G, spec, expected_model_id=mid, graph_level_weather=derived or None))
                per_model.setdefault(mid, {})[d.name] = G
            report.add(nx_checks.check_graph_collection(list(graphs.values()), spec))
        for mid, gs in per_model.items():
            if len(gs) > 1:
                report.add(cross_config_checks.check_graphs_consistent(gs, model_id=mid))

    # ---- 4. PyG ----------------------------------------------------------------------
    if not args.skip_pyg:
        from beg.validation import pyg_checks

        if not pyg_checks.torch_available():
            report.add(CheckResult.skip("pyg.available", "torch / torch_geometric not importable", scope="pyg"))
        else:
            pyg_root = Path(args.pyg_graphs) if args.pyg_graphs else paths.pyg_graphs_root
            for d in list_pyg_config_dirs(pyg_root):
                spec, derived = resolve_spec_for_output_name(d.name, all_specs)
                if spec is None or (args.config and spec.name not in args.config):
                    continue
                graphs, ids = [], []
                for mid, path in iter_pyg_graph_files(d, limit=args.max_models):
                    g = pyg_checks.load_graph_file(path)
                    graphs.append(g)
                    ids.append(mid)
                    kw = {"model_id": mid}
                    if derived:
                        kw.update(graph_feat_key="weather_city", graph_feat_dim=223)
                    if spec.homogeneous:
                        kw["expect_symmetric"] = all(e.bidirectional for e in spec.edges)
                    report.add(pyg_checks.check_graph(g, spec, **kw))
                if graphs:
                    _log(f"pyg checks: {d.name} ({len(graphs)} graphs)")
                    report.add(pyg_checks.check_graph_collection(graphs, spec, model_ids=ids))
            ds_root = Path(args.pyg_datasets) if args.pyg_datasets else paths.pyg_datasets_root
            tr_payloads = {}
            for name, (pt, meta) in list_training_ready(ds_root).items():
                spec, _ = resolve_spec_for_output_name(name, all_specs)
                if args.config and (spec is None or spec.name not in args.config):
                    continue
                _log(f"training_ready checks: {name}")
                payload = pyg_checks.load_graph_file(pt)
                tr_payloads[name] = payload
                report.add(pyg_checks.check_training_ready_payload(payload, spec, meta=load_meta_json(meta), max_graphs_checked=args.max_models))
            if len(tr_payloads) > 1:
                report.add(pyg_checks.check_training_ready_alignment(tr_payloads))

    # ---- 5. labels -------------------------------------------------------------------
    labels_csv = Path(args.labels) if args.labels else paths.labels_csv
    if labels_csv.exists():
        ids = set()
        for p in payloads.values():
            ids.update(map(str, p.get("model_ids") or []))
        report.add(label_checks.check_labels(label_checks.load_labels(labels_csv), dataset_model_ids=ids or None))

    # ---- report ----------------------------------------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else paths.outputs / "graph_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "graph_quality_report.md"
    json_path = out_dir / "graph_quality_report.json"
    md_path.write_text(report.to_markdown(include_pass=args.include_pass), encoding="utf-8")
    report.to_json(str(json_path))
    counts = report.counts()
    print()
    print(f"Summary: {counts}")
    for r in report.failures[:40]:
        print("  ", r)
    if len(report.failures) > 40:
        print(f"   … {len(report.failures) - 40} more failures")
    print(f"Report written to {md_path} and {json_path}")
    if report.failures:
        return 1
    if args.fail_on_warn and report.warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
