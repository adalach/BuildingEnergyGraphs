"""Reading epJSON files into the generated ``ep`` tables.

One transaction per file. Within it the model is registered in ``ep_model``,
every object instance becomes a main row, and array-of-objects fields become
child rows. A file whose SHA-256 is already registered is skipped or upserted
according to the duplicate policy.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import text as sa_text

from beg.ingest.ep_schema import EnergyPlusSchema
from beg.ingest.foreign_keys_advanced import fully_qualified
from beg.ingest.models import (
    build_scalar_field_map,
    child_rows_for,
    count_instances_in_container,
    insert_main_rows_batched,
    insert_model,
    insert_or_upsert_main,
    load_epjson,
    normalize_instances_container,
    replace_children,
    row_data_from_fields,
    sha256_file,
)
from beg.ingest.tables import SchemaBuild


@dataclass
class IngestResult:
    """Counts and timings from one ingestion run."""

    files_total: int = 0
    files_new: int = 0
    files_duplicate: int = 0
    policy: str = "skip"
    seconds_total: float = 0.0
    seconds_per_file: float = 0.0
    table_counts: dict = field(repr=False, default_factory=lambda: defaultdict(int))
    object_main_counts: Counter = field(repr=False, default_factory=Counter)
    child_field_counts: Counter = field(repr=False, default_factory=Counter)
    skipped_objects: Counter = field(repr=False, default_factory=Counter)
    files_report: list = field(repr=False, default_factory=list)

    @property
    def rows_main(self) -> int:
        return sum(self.object_main_counts.values())

    @property
    def rows_child(self) -> int:
        return sum(self.child_field_counts.values())

    def as_dict(self) -> dict:
        """The scalar fields only, for the run summary JSON."""
        return {
            "files_total": self.files_total,
            "files_new": self.files_new,
            "files_duplicate": self.files_duplicate,
            "policy": self.policy,
            "seconds_total": self.seconds_total,
            "seconds_per_file": self.seconds_per_file,
            "rows_main": self.rows_main,
            "rows_child": self.rows_child,
        }

    def summary(self) -> pd.DataFrame:
        """One-row frame of the headline numbers."""
        return pd.DataFrame([self.as_dict()])

    def top_objects(self, n: int = 10) -> pd.DataFrame:
        """Object types by number of main rows written."""
        return pd.DataFrame(
            self.object_main_counts.most_common(n), columns=["object", "main_rows"]
        )

    def top_child_fields(self, n: int = 10) -> pd.DataFrame:
        """Array fields by number of child rows written."""
        return pd.DataFrame(
            self.child_field_counts.most_common(n), columns=["object.field", "child_rows"]
        )

    def skipped_frame(self) -> pd.DataFrame:
        """Object types present in the models but with no table defined."""
        return pd.DataFrame(
            self.skipped_objects.most_common(), columns=["object", "instances_skipped"]
        )


def dataset_already_ingested(engine, epjson_files, schema: str) -> tuple[bool, dict]:
    """Is every current file's hash already in ``ep_model``?

    A restart optimisation, not an integrity audit: it compares file hashes
    against the registry and says nothing about whether the object tables are
    complete.
    """
    files_total = len(epjson_files)
    if files_total == 0:
        return False, {"files_total": 0, "db_rows": 0, "matched_hashes": 0, "missing_hashes": 0}

    fq_ep_model = fully_qualified(schema, "ep_model")
    with engine.connect() as con:
        table_exists = con.execute(
            sa_text(
                """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = 'ep_model'
            )
        """
            ),
            {"schema": schema},
        ).scalar()
        if not table_exists:
            return False, {
                "files_total": files_total,
                "db_rows": 0,
                "matched_hashes": 0,
                "missing_hashes": files_total,
            }

        db_rows = int(con.execute(sa_text(f"SELECT COUNT(*) FROM {fq_ep_model}")).scalar() or 0)
        db_hashes = {
            row[0]
            for row in con.execute(
                sa_text(f"SELECT sha256 FROM {fq_ep_model} WHERE sha256 IS NOT NULL")
            ).fetchall()
            if row[0]
        }

    file_hashes = [sha256_file(Path(fp)) for fp in epjson_files]
    matched = sum(1 for h in file_hashes if h in db_hashes)
    missing = files_total - matched

    return (missing == 0 and db_rows >= files_total), {
        "files_total": files_total,
        "db_rows": db_rows,
        "matched_hashes": matched,
        "missing_hashes": missing,
    }


#: Roughly how many progress lines a run should emit, whatever its length.
PROGRESS_UPDATES = 40


def _progress_iterator(epjson_files, progress):
    """Wrap the file list according to ``progress``. Returns ``(iterable, callback)``."""
    if progress is True:
        # Plain-text tqdm, not tqdm.auto: it needs no ipywidgets, and a text bar
        # survives in saved notebook output.
        from tqdm import tqdm

        # Every refresh is stored in the notebook output, so the bar steps by
        # iterations, not time: about PROGRESS_UPDATES lines whatever the run length.
        total = len(epjson_files)
        return (
            tqdm(
                epjson_files,
                desc="Ingesting epJSON",
                unit="model",
                file=sys.stdout,
                miniters=max(1, total // PROGRESS_UPDATES),
                mininterval=0,
            ),
            None,
        )
    if callable(progress):
        return epjson_files, progress
    return epjson_files, None


def ingest_models(
    engine,
    epjson_files,
    build: SchemaBuild,
    schema: EnergyPlusSchema,
    duplicate_policy: str = "skip",
    progress=None,
) -> IngestResult:
    """Ingest every epJSON file into the tables defined by ``build``.

    ``duplicate_policy`` is ``"skip"`` (leave an already-registered file alone)
    or ``"update"`` (re-read it and upsert its rows).

    ``progress`` is ``True`` for a tqdm progress bar, a ``callable(done, total,
    filename)`` for custom reporting, or ``None`` for silence.
    """
    result = IngestResult(files_total=len(epjson_files), policy=duplicate_policy)
    iterator, callback = _progress_iterator(epjson_files, progress)
    field_maps: dict[str, dict] = {}  # per object type; identical for every file
    t0 = time.perf_counter()

    for done, fp in enumerate(iterator, start=1):
        file_start = time.perf_counter()
        data = load_epjson(fp)
        file_main_rows = 0
        file_child_rows = 0

        with engine.begin() as con:
            model_id, is_new = insert_model(con, Path(fp), build.ep_model)

            if is_new:
                status = "new"
                result.files_new += 1
            else:
                status = "duplicate_update" if duplicate_policy == "update" else "duplicate_skip"
                result.files_duplicate += 1

                if duplicate_policy == "skip":
                    result.files_report.append(
                        {
                            "file": Path(fp).name,
                            "status": status,
                            "seconds": round(time.perf_counter() - file_start, 3),
                            "new_rows_main": 0,
                            "new_rows_child": 0,
                        }
                    )
                    if callback is not None:
                        callback(done, result.files_total, Path(fp).name)
                    continue

            for obj_name, instances in data.items():
                if obj_name not in build.table_map:
                    result.skipped_objects[obj_name] += count_instances_in_container(instances)
                    continue

                main_tbl, child_obj_tbls = build.table_map[obj_name]
                field_map = field_maps.get(obj_name)
                if field_map is None:
                    field_map = field_maps[obj_name] = build_scalar_field_map(
                        obj_name, schema.tables, build.scalar_col_map
                    )

                if is_new:
                    # Batched path: one statement for the main rows of an object
                    # type, then one per array field for its child rows.
                    instance_fields = []
                    main_rows = []
                    for inst_name, fields in normalize_instances_container(instances):
                        instance_fields.append(fields)
                        main_rows.append(
                            row_data_from_fields(model_id, inst_name, field_map, fields)
                        )

                    parent_ids = insert_main_rows_batched(con, main_tbl, main_rows)

                    n_main = len(parent_ids)
                    result.table_counts[main_tbl.name] += n_main
                    result.object_main_counts[obj_name] += n_main
                    file_main_rows += n_main

                    for child_field, child_tbl in child_obj_tbls.items():
                        child_rows = child_rows_for(
                            obj_name, child_field, parent_ids, instance_fields,
                            build.object_child_colmap,
                        )
                        if not child_rows:
                            continue
                        con.execute(child_tbl.insert(), child_rows)
                        n_child = len(child_rows)
                        file_child_rows += n_child
                        result.child_field_counts[f"{obj_name}.{child_field}"] += n_child
                        result.table_counts[child_tbl.name] += n_child
                    continue

                # Upsert path: a conflict has to be resolved per row, so this is
                # one statement per instance.
                for inst_name, fields in normalize_instances_container(instances):
                    row_data = row_data_from_fields(model_id, inst_name, field_map, fields)
                    parent_id = insert_or_upsert_main(
                        con, main_tbl, row_data, is_new, duplicate_policy
                    )

                    result.table_counts[main_tbl.name] += 1
                    result.object_main_counts[obj_name] += 1
                    file_main_rows += 1

                    inserted, per_field = replace_children(
                        con, obj_name, fields, parent_id, child_obj_tbls,
                        build.object_child_colmap, parent_is_new=is_new,
                    )
                    file_child_rows += inserted
                    result.child_field_counts.update(per_field)

                    for compound_key, n in per_field.items():
                        child_field = compound_key.split(".", 1)[1]
                        result.table_counts[child_obj_tbls[child_field].name] += n

        result.files_report.append(
            {
                "file": Path(fp).name,
                "status": status,
                "seconds": round(time.perf_counter() - file_start, 3),
                "new_rows_main": file_main_rows,
                "new_rows_child": file_child_rows,
            }
        )
        if callback is not None:
            callback(done, result.files_total, Path(fp).name)

    result.seconds_total = round(time.perf_counter() - t0, 3)
    result.seconds_per_file = (
        round(result.seconds_total / result.files_total, 3) if result.files_total else 0.0
    )
    return result
