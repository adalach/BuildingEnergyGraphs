"""Surface shading proxy metrics.

Populates ``<schema>.surface_shading_metrics``, a precondition for any
configuration whose transform list includes ``surface_shading_metrics.sql``.
:func:`beg.pipeline.sql.ensure_required_source_tables` calls this automatically
when the table is missing or incomplete; all transforms are then rebuilt.

Also runnable directly::

    python -m beg.pipeline.shading --replace
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from sqlalchemy import inspect, text

from beg.db import get_engine
from beg.db.utils import ensure_standard_relation_indexes


@dataclass
class WallSurface:
    surface_id: int
    model_id: str
    instance_name: str
    zone_name: str | None
    line_2d: LineString
    midpoint: Point
    outward_normal: tuple[float, float]
    z_center: float | None


@dataclass
class ShadeSurface:
    shade_id: int
    model_id: str
    instance_name: str
    line_2d: LineString
    z_max: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate ep.surface_shading_metrics from EP wall/shade geometry."
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Override the configured BEG_DSN. Normally omitted.",
    )
    parser.add_argument("--schema", default="ep")
    parser.add_argument("--replace", action="store_true", help="Drop and rebuild the table.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model_ids already present in the output table.",
    )
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--batch-models", type=int, default=100)
    parser.add_argument("--interval-m", type=float, default=5.0)
    parser.add_argument("--sample-cap", type=int, default=5)
    parser.add_argument("--n-closest", type=int, default=3)
    parser.add_argument("--outward-eps", type=float, default=0.2)
    parser.add_argument("--max-dist", type=float, default=None)
    parser.add_argument("--max-angle-deg", type=float, default=85.0)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    return parser.parse_args()


def _build_placeholders(values: Sequence[str], prefix: str = "m") -> tuple[str, dict[str, str]]:
    placeholders = ",".join(f":{prefix}{i}" for i in range(len(values)))
    params = {f"{prefix}{i}": value for i, value in enumerate(values)}
    return placeholders, params


def _unique_xy_pairs(vertices: Sequence[tuple[float, float, float]], precision: int = 6) -> list[tuple[float, float]]:
    seen: dict[tuple[float, float], tuple[float, float]] = {}
    for x, y, _ in vertices:
        key = (round(float(x), precision), round(float(y), precision))
        seen.setdefault(key, (float(x), float(y)))
    return list(seen.values())


def _segment_from_vertices(vertices: Sequence[tuple[float, float, float]]) -> LineString | None:
    points = _unique_xy_pairs(vertices)
    if len(points) < 2:
        return None
    if len(points) == 2:
        return LineString(points)

    best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_dist = -1.0
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist > best_dist:
                best_dist = dist
                best_pair = (points[i], points[j])
    if best_pair is None or best_dist <= 0:
        return None
    return LineString(best_pair)


def _z_stats(vertices: Sequence[tuple[float, float, float]]) -> tuple[float | None, float | None, float | None]:
    zs = [float(z) for _, _, z in vertices if z is not None]
    if not zs:
        return None, None, None
    z_min = min(zs)
    z_max = max(zs)
    return z_min, z_max, (z_min + z_max) / 2.0


def _unit(vec: tuple[float, float]) -> tuple[float, float] | None:
    norm = math.hypot(vec[0], vec[1])
    if norm == 0:
      return None
    return vec[0] / norm, vec[1] / norm


def _wall_outward_normal(line: LineString, building_centroid: Point) -> tuple[float, float] | None:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    n1 = _unit((-dy, dx))
    n2 = _unit((dy, -dx))
    if n1 is None or n2 is None:
        return None
    midpoint = line.interpolate(0.5, normalized=True)
    ref = (midpoint.x - building_centroid.x, midpoint.y - building_centroid.y)
    return n1 if (n1[0] * ref[0] + n1[1] * ref[1]) >= (n2[0] * ref[0] + n2[1] * ref[1]) else n2


def _sample_points(line: LineString, interval_m: float, sample_cap: int) -> list[Point]:
    if line.length <= 0:
        return []
    n_segments = max(int(math.ceil(line.length / max(interval_m, 1e-6))), 1)
    n_segments = min(n_segments, max(sample_cap, 1))
    return [
        line.interpolate((i / n_segments) * line.length)
        for i in range(n_segments + 1)
    ]


def _is_blocked(
    point: Point,
    target: Point,
    outward_normal: tuple[float, float],
    blockers: Sequence[LineString],
    own_idx: int,
    outward_eps: float,
) -> bool:
    start = Point(
        point.x + outward_normal[0] * outward_eps,
        point.y + outward_normal[1] * outward_eps,
    )
    ray = LineString([(start.x, start.y), (target.x, target.y)])
    for idx, blocker in enumerate(blockers):
        if idx == own_idx or blocker.is_empty:
            continue
        if ray.crosses(blocker):
            return True
    return False


def _fetch_model_ids(engine, schema: str, limit: int | None) -> list[str]:
    sql = text(
        f"""
        SELECT DISTINCT s.model_id
        FROM {schema}.ep_buildingsurface_detailed s
        JOIN {schema}.ep_shading_building_detailed sh
          ON sh.model_id = s.model_id
        WHERE s.outside_boundary_condition_outdoors = 1
          AND s.surface_type_wall = 1
        ORDER BY s.model_id
        """
    )
    with engine.connect() as con:
        model_ids = pd.read_sql(sql, con)["model_id"].astype(str).tolist()
    return model_ids[:limit] if limit is not None else model_ids


def _fetch_processed_model_ids(engine, schema: str) -> set[str]:
    inspector = inspect(engine)
    if not inspector.has_table("surface_shading_metrics", schema=schema):
        return set()
    with engine.connect() as con:
        df = pd.read_sql(
            text(f"SELECT DISTINCT model_id FROM {schema}.surface_shading_metrics"),
            con,
        )
    return set(df["model_id"].astype(str).tolist())


def _fetch_wall_vertices(engine, schema: str, model_ids: Sequence[str]) -> pd.DataFrame:
    placeholders, params = _build_placeholders(model_ids)
    sql = text(
        f"""
        SELECT
          s.id AS surface_id,
          s.model_id,
          s.instance_name,
          s.zone_name,
          v.position,
          v.vertex_x_coordinate AS x,
          v.vertex_y_coordinate AS y,
          v.vertex_z_coordinate AS z
        FROM {schema}.ep_buildingsurface_detailed s
        JOIN {schema}.ep_buildingsurface_detailed_vertices v
          ON v.parent_id = s.id
        WHERE s.model_id IN ({placeholders})
          AND s.outside_boundary_condition_outdoors = 1
          AND s.surface_type_wall = 1
        ORDER BY s.model_id, s.id, v.position
        """
    )
    with engine.connect() as con:
        return pd.read_sql(sql, con, params=params)


def _fetch_shade_vertices(engine, schema: str, model_ids: Sequence[str]) -> pd.DataFrame:
    placeholders, params = _build_placeholders(model_ids)
    sql = text(
        f"""
        SELECT
          s.id AS shade_id,
          s.model_id,
          s.instance_name,
          v.position,
          v.vertex_x_coordinate AS x,
          v.vertex_y_coordinate AS y,
          v.vertex_z_coordinate AS z
        FROM {schema}.ep_shading_building_detailed s
        JOIN {schema}.ep_shading_building_detailed_vertices v
          ON v.parent_id = s.id
        WHERE s.model_id IN ({placeholders})
        ORDER BY s.model_id, s.id, v.position
        """
    )
    with engine.connect() as con:
        return pd.read_sql(sql, con, params=params)


def _build_walls_for_model(model_id: str, wall_vertices: pd.DataFrame) -> list[WallSurface]:
    rows: list[tuple[int, str, str | None, list[tuple[float, float, float]]]] = []
    for surface_id, grp in wall_vertices.groupby("surface_id", sort=False):
        grp = grp.sort_values("position")
        vertices = list(zip(grp["x"], grp["y"], grp["z"]))
        rows.append(
            (
                int(surface_id),
                str(grp["instance_name"].iloc[0]),
                None if pd.isna(grp["zone_name"].iloc[0]) else str(grp["zone_name"].iloc[0]),
                vertices,
            )
        )
    endpoints: list[tuple[float, float]] = []
    temp_lines: list[tuple[int, str, str | None, LineString, float | None]] = []
    for surface_id, instance_name, zone_name, vertices in rows:
        line = _segment_from_vertices(vertices)
        if line is None or line.length <= 0:
            continue
        _, _, z_center = _z_stats(vertices)
        temp_lines.append((surface_id, instance_name, zone_name, line, z_center))
        endpoints.extend(list(line.coords))
    if not temp_lines or not endpoints:
        return []
    building_centroid = MultiPoint(endpoints).convex_hull.centroid
    walls: list[WallSurface] = []
    for surface_id, instance_name, zone_name, line, z_center in temp_lines:
        outward_normal = _wall_outward_normal(line, building_centroid)
        if outward_normal is None:
            continue
        walls.append(
            WallSurface(
                surface_id=surface_id,
                model_id=model_id,
                instance_name=instance_name,
                zone_name=zone_name,
                line_2d=line,
                midpoint=line.interpolate(0.5, normalized=True),
                outward_normal=outward_normal,
                z_center=z_center,
            )
        )
    return walls


def _build_shades_for_model(model_id: str, shade_vertices: pd.DataFrame) -> list[ShadeSurface]:
    shades: list[ShadeSurface] = []
    for shade_id, grp in shade_vertices.groupby("shade_id", sort=False):
        grp = grp.sort_values("position")
        vertices = list(zip(grp["x"], grp["y"], grp["z"]))
        line = _segment_from_vertices(vertices)
        if line is None or line.length <= 0:
            continue
        _, z_max, _ = _z_stats(vertices)
        shades.append(
            ShadeSurface(
                shade_id=int(shade_id),
                model_id=model_id,
                instance_name=str(grp["instance_name"].iloc[0]),
                line_2d=line,
                z_max=z_max,
            )
        )
    return shades


def _process_model(
    model_id: str,
    walls: Sequence[WallSurface],
    shades: Sequence[ShadeSurface],
    interval_m: float,
    sample_cap: int,
    n_closest: int,
    outward_eps: float,
    max_dist: float | None,
    max_angle_deg: float,
) -> pd.DataFrame:
    if not walls or not shades:
        return pd.DataFrame()

    blocker_lines = [w.line_2d for w in walls]
    angle_threshold = math.radians(max_angle_deg)
    records: list[dict[str, float | int | str | None]] = []
    wall_meta = pd.DataFrame(
        [
            {
                "model_id": wall.model_id,
                "surface_id": wall.surface_id,
                "buildingsurface_id": wall.instance_name,
                "zone_id": wall.zone_name,
            }
            for wall in walls
        ]
    )

    for own_idx, wall in enumerate(walls):
        points = _sample_points(wall.line_2d, interval_m=interval_m, sample_cap=sample_cap)
        for point in points:
            candidates = []
            for shade in shades:
                nearest_on_shade = shade.line_2d.interpolate(shade.line_2d.project(point))
                dx = nearest_on_shade.x - point.x
                dy = nearest_on_shade.y - point.y
                dist = math.hypot(dx, dy)
                if dist <= 1e-9:
                    continue
                if max_dist is not None and dist > max_dist:
                    continue
                ray_unit = (dx / dist, dy / dist)
                dot = ray_unit[0] * wall.outward_normal[0] + ray_unit[1] * wall.outward_normal[1]
                dot = max(min(dot, 1.0), -1.0)
                angle = math.acos(dot)
                if angle > angle_threshold:
                    continue
                if _is_blocked(
                    point=point,
                    target=nearest_on_shade,
                    outward_normal=wall.outward_normal,
                    blockers=blocker_lines,
                    own_idx=own_idx,
                    outward_eps=outward_eps,
                ):
                    continue
                candidates.append((dist, dx, dy, shade))

            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            for dist, dx, dy, shade in candidates[: max(n_closest, 1)]:
                records.append(
                    {
                        "model_id": model_id,
                        "surface_id": wall.surface_id,
                        "buildingsurface_id": wall.instance_name,
                        "zone_id": wall.zone_name,
                        "shade_id": shade.shade_id,
                        "dist": dist,
                        "dx": dx,
                        "dy": dy,
                        "surface_z_center": wall.z_center,
                        "shade_z_max": shade.z_max,
                    }
                )

    if not records:
        zero_metrics = wall_meta.copy()
        zero_metrics["shade_count"] = 0
        zero_metrics["ray_count"] = 0
        zero_metrics["min_dist"] = 0.0
        zero_metrics["mean_dist"] = 0.0
        zero_metrics["p90_dist"] = 0.0
        zero_metrics["ray_dir_mean_sin"] = 0.0
        zero_metrics["ray_dir_mean_cos"] = 0.0
        zero_metrics["ray_elevation_mean_rad"] = 0.0
        zero_metrics["ray_elevation_mean_sin"] = 0.0
        zero_metrics["ray_elevation_mean_cos"] = 0.0
        return zero_metrics[
            [
                "model_id",
                "buildingsurface_id",
                "zone_id",
                "shade_count",
                "ray_count",
                "min_dist",
                "mean_dist",
                "p90_dist",
                "ray_dir_mean_sin",
                "ray_dir_mean_cos",
                "ray_elevation_mean_rad",
                "ray_elevation_mean_sin",
                "ray_elevation_mean_cos",
            ]
        ]

    edges = pd.DataFrame.from_records(records)
    edges["ray_azimuth_rad"] = np.arctan2(edges["dx"], edges["dy"])
    edges["ray_dir_mean_sin"] = np.sin(edges["ray_azimuth_rad"])
    edges["ray_dir_mean_cos"] = np.cos(edges["ray_azimuth_rad"])

    shortest = edges.sort_values("dist").groupby(["model_id", "surface_id", "shade_id"], as_index=False).first()
    shortest["ray_elevation_mean_rad"] = np.arctan2(
        shortest["shade_z_max"] - shortest["surface_z_center"],
        shortest["dist"],
    )
    shortest["ray_elevation_mean_sin"] = np.sin(shortest["ray_elevation_mean_rad"])
    shortest["ray_elevation_mean_cos"] = np.cos(shortest["ray_elevation_mean_rad"])

    metrics = pd.concat(
        [
            shortest.groupby(["model_id", "surface_id"])["shade_id"].nunique().rename("shade_count"),
            edges.groupby(["model_id", "surface_id"]).size().rename("ray_count"),
            edges.groupby(["model_id", "surface_id"])["dist"].min().rename("min_dist"),
            edges.groupby(["model_id", "surface_id"])["dist"].mean().rename("mean_dist"),
            edges.groupby(["model_id", "surface_id"])["dist"].quantile(0.9).rename("p90_dist"),
            edges.groupby(["model_id", "surface_id"])["ray_dir_mean_sin"].mean().rename("ray_dir_mean_sin"),
            edges.groupby(["model_id", "surface_id"])["ray_dir_mean_cos"].mean().rename("ray_dir_mean_cos"),
            shortest.groupby(["model_id", "surface_id"])["ray_elevation_mean_rad"].mean().rename("ray_elevation_mean_rad"),
            shortest.groupby(["model_id", "surface_id"])["ray_elevation_mean_sin"].mean().rename("ray_elevation_mean_sin"),
            shortest.groupby(["model_id", "surface_id"])["ray_elevation_mean_cos"].mean().rename("ray_elevation_mean_cos"),
        ],
        axis=1,
    ).reset_index()

    metrics = wall_meta.merge(metrics, on=["model_id", "surface_id"], how="left")
    fill_zero_cols = [
        "shade_count",
        "ray_count",
        "min_dist",
        "mean_dist",
        "p90_dist",
        "ray_dir_mean_sin",
        "ray_dir_mean_cos",
        "ray_elevation_mean_rad",
        "ray_elevation_mean_sin",
        "ray_elevation_mean_cos",
    ]
    metrics[fill_zero_cols] = metrics[fill_zero_cols].fillna(0)
    metrics = metrics[
        [
            "model_id",
            "buildingsurface_id",
            "zone_id",
            "shade_count",
            "ray_count",
            "min_dist",
            "mean_dist",
            "p90_dist",
            "ray_dir_mean_sin",
            "ray_dir_mean_cos",
            "ray_elevation_mean_rad",
            "ray_elevation_mean_sin",
            "ray_elevation_mean_cos",
        ]
    ]
    return metrics


def _process_model_payload(payload: tuple) -> pd.DataFrame:
    (
        model_id,
        wall_df,
        shade_df,
        interval_m,
        sample_cap,
        n_closest,
        outward_eps,
        max_dist,
        max_angle_deg,
    ) = payload
    walls = _build_walls_for_model(model_id, wall_df)
    shades = _build_shades_for_model(model_id, shade_df)
    return _process_model(
        model_id=model_id,
        walls=walls,
        shades=shades,
        interval_m=interval_m,
        sample_cap=sample_cap,
        n_closest=n_closest,
        outward_eps=outward_eps,
        max_dist=max_dist,
        max_angle_deg=max_angle_deg,
    )


def _ensure_output_table(engine, schema: str, replace: bool) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema}.surface_shading_metrics (
      model_id TEXT NOT NULL,
      buildingsurface_id TEXT NOT NULL,
      zone_id TEXT NULL,
      shade_count INTEGER NULL,
      ray_count INTEGER NULL,
      min_dist DOUBLE PRECISION NULL,
      mean_dist DOUBLE PRECISION NULL,
      p90_dist DOUBLE PRECISION NULL,
      ray_dir_mean_sin DOUBLE PRECISION NULL,
      ray_dir_mean_cos DOUBLE PRECISION NULL,
      ray_elevation_mean_rad DOUBLE PRECISION NULL,
      ray_elevation_mean_sin DOUBLE PRECISION NULL,
      ray_elevation_mean_cos DOUBLE PRECISION NULL
    )
    """
    with engine.begin() as con:
        if replace:
            con.execute(text(f"DROP TABLE IF EXISTS {schema}.surface_shading_metrics"))
        con.execute(text(ddl))


def _write_batch(engine, schema: str, batch_df: pd.DataFrame) -> None:
    if batch_df.empty:
        return
    batch_df.to_sql(
        "surface_shading_metrics",
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )


def _finalize_output(engine, schema: str) -> None:
    ensure_standard_relation_indexes(engine, schema, "surface_shading_metrics", verbose=False)
    with engine.begin() as con:
        con.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_surface_shading_metrics_model_id "
                f"ON {schema}.surface_shading_metrics (model_id)"
            )
        )
        con.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_surface_shading_metrics_model_surface "
                f"ON {schema}.surface_shading_metrics (model_id, buildingsurface_id)"
            )
        )


def populate_surface_shading_metrics(
    engine,
    *,
    schema: str = "ep",
    replace: bool = False,
    resume: bool = False,
    model_limit: int | None = None,
    batch_models: int = 100,
    interval_m: float = 5.0,
    sample_cap: int = 5,
    n_closest: int = 3,
    outward_eps: float = 0.2,
    max_dist: float | None = None,
    max_angle_deg: float = 85.0,
    workers: int | None = None,
) -> int:
    """Populate the canonical surface shading table and return its row count."""
    if workers is None:
        workers = max((os.cpu_count() or 2) - 1, 1)

    model_ids = _fetch_model_ids(engine, schema, limit=model_limit)
    if resume:
        processed = _fetch_processed_model_ids(engine, schema)
        model_ids = [mid for mid in model_ids if mid not in processed]

    print(f"Models to process: {len(model_ids)}")
    _ensure_output_table(engine, schema, replace=replace)
    if not model_ids:
        print("Nothing to do.")
        _finalize_output(engine, schema)
        with engine.connect() as con:
            return int(
                con.execute(text(f"SELECT COUNT(*) FROM {schema}.surface_shading_metrics")).scalar_one()
            )

    total_rows = 0
    started = time.time()
    report_every = max(len(model_ids) // 10, 1)
    next_report = report_every
    for start_idx in range(0, len(model_ids), batch_models):
        batch_model_ids = model_ids[start_idx : start_idx + batch_models]
        wall_vertices = _fetch_wall_vertices(engine, schema, batch_model_ids)
        shade_vertices = _fetch_shade_vertices(engine, schema, batch_model_ids)
        grouped_walls = {str(mid): grp.copy() for mid, grp in wall_vertices.groupby("model_id", sort=False)}
        grouped_shades = {str(mid): grp.copy() for mid, grp in shade_vertices.groupby("model_id", sort=False)}
        payloads = [
            (
                model_id,
                grouped_walls.get(model_id, wall_vertices.iloc[0:0].copy()),
                grouped_shades.get(model_id, shade_vertices.iloc[0:0].copy()),
                interval_m,
                sample_cap,
                n_closest,
                outward_eps,
                max_dist,
                max_angle_deg,
            )
            for model_id in batch_model_ids
        ]

        if workers <= 1:
            batch_frames = [df for df in map(_process_model_payload, payloads) if not df.empty]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                batch_frames = [df for df in pool.map(_process_model_payload, payloads) if not df.empty]

        batch_out = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame()
        _write_batch(engine, schema, batch_out)
        total_rows += len(batch_out)
        elapsed = time.time() - started
        done = min(start_idx + len(batch_model_ids), len(model_ids))
        if done >= next_report or done == len(model_ids):
            print(f"[{done}/{len(model_ids)}] total_rows={total_rows} elapsed={elapsed/60.0:.1f} min")
            next_report = done + report_every

    _finalize_output(engine, schema)
    print("Done.")
    with engine.connect() as con:
        final_rows = int(
            con.execute(text(f"SELECT COUNT(*) FROM {schema}.surface_shading_metrics")).scalar_one()
        )
    print(f"Final rows in {schema}.surface_shading_metrics: {final_rows}")
    return final_rows


def main() -> None:
    args = parse_args()
    engine = get_engine(args.dsn)
    populate_surface_shading_metrics(
        engine,
        schema=args.schema,
        replace=args.replace,
        resume=args.resume,
        model_limit=args.model_limit,
        batch_models=args.batch_models,
        interval_m=args.interval_m,
        sample_cap=args.sample_cap,
        n_closest=args.n_closest,
        outward_eps=args.outward_eps,
        max_dist=args.max_dist,
        max_angle_deg=args.max_angle_deg,
        workers=args.workers,
    )
    print("Important next step:")
    print("  Rebuild the transforms and graph caches so shading features propagate into the comp_*/zones/building tables.")
    print("  In notebooks/BEG_02_Graphs.ipynb set:")
    print("    REBUILD_TRANSFORMS = True")
    print("    FORCE_REBUILD_DATAFRAMES = True")
    print("    FORCE_REBUILD_PYG_GRAPHS = True")


if __name__ == "__main__":
    main()
