"""Compute the per-reach feature rows the static forecast scores, one file per reach.

This is the second half of the offline forecast build. widget/static/fetch_basins.py gets the basin
polygons from NLDI; this turns each into the feature row a browser needs, using the same
recipes._agg_block the training path uses so the two cannot drift.

    python -m widget.static.build_reaches --status       # progress, computes nothing
    python -m widget.static.build_reaches                # everything still missing
    python -m widget.static.build_reaches --limit 200    # one bounded batch
    python -m widget.static.build_reaches --workers 6    # parallel (this is CPU-bound)

TWO CACHES, SPLIT AT THE COST. GRIDS holds the clipped cells -- a D8 flow-distance field plus a cell intersection, the hours in this pass -- and depends on the basin and grid_global alone, so a retuned recipe reuses them. ROWS holds the aggregation over those cells, keyed by ROW_SCHEMA and the recipe fingerprint, and rebuilds at ~140 ms a reach -- 40 min for the cohort single-threaded, minutes at --workers 6. Changing the SHIPPED format is a re-pack (seconds), a third step again.

Ctrl-C is safe: each row is written whole via a temp file and renamed, and a restart skips what exists. Reaches whose basin intersects no grid cell -- real, for the smallest ones -- are recorded as tombstones rather than retried forever.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # widget/static
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from widget.static.build_bundle import FORECAST_YEARS  # noqa: E402
from widget.static.fetch_basins import CACHE as BASIN_CACHE, reach_ids  # noqa: E402

ROWS = _ROOT / "src" / "data" / "cache" / "reach_rows"
FAILED = ROWS / "failed.json"

# The basin's clipped cells, cached apart from the row built out of them: they cost the pass its hours and no recipe setting touches them, so a retune re-aggregates instead of re-delineating. GRID_SCHEMA guards what does invalidate them -- grid_global, or the D8 raster.
GRIDS = _ROOT / "src" / "data" / "cache" / "reach_grids"
GRID_SCHEMA = 1

_STATE = {}  # per-process lazies: the flowline outlets and the weather basis


def _state():
    """Load the flowlines and weather basis once per worker process."""
    if not _STATE:
        import geopandas as gpd

        from widget.static import build_forecast as bf
        from src.features import recipes

        fl = gpd.read_parquet(
            _ROOT / "src/data/processed/map_overlays/iowa_flowlines.parquet", columns=["COMID", "geometry"]
        ).set_index("COMID")
        _STATE["outlets"] = {int(c): tuple(g.coords)[-1][:2] for c, g in fl["geometry"].items()}  # (lon, lat)
        _STATE["basis"] = bf.load_weather_basis(_ROOT / "src" / "data" / "cache")
        _STATE["tasks"] = {
            t: (recipes.LIGHT_EDGES[t], recipes.LIGHT_VEL[t], recipes.LIGHT_LAM[t]) for t in ("reg", "clf")
        }
        _STATE["bf"] = bf
    return _STATE


def _is_current(path: Path) -> bool:
    """Whether a cached row matches the current reach_row SHAPE and the current recipe CONTENT.

    Two separate ways a cached row goes stale, and both are silent: ROW_SCHEMA changes when the
    output structure does, recipe_fingerprint changes when a bucket edge, decay length or weather
    geometry is retuned -- which leaves the structure intact and every value in it wrong.
    """
    from widget.static.build_forecast import ROW_SCHEMA, recipe_fingerprint

    try:
        row = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 -- unreadable or truncated: treat as needing a rebuild
        return False
    return row.get("schema") == ROW_SCHEMA and row.get("recipe") == recipe_fingerprint()


def _repair(basin):
    """make_valid a self-intersecting NLDI polygon; pass a valid one through.

    A few come back self-intersecting and raise GEOSException against the grid. That is a property of the cached file, not a transient, so without this the reach fails identically on every retry and never gets a row.
    """
    from shapely import make_valid

    if basin.geometry.is_valid.all():
        return basin
    out = basin.copy()
    out["geometry"] = [g if g.is_valid else make_valid(g) for g in out.geometry]
    return out


def _cells(comid: int, basin_path: Path, lat: float, lon: float):
    """The basin's clipped cells, cached (see GRIDS). None when nothing intersects.

    build_site_view is the D8 field plus the cell clip -- the pass's whole cost. Geometry is dropped on the way to disk: it exists only to produce frac_cell_in_basin.
    """
    import pandas as pd

    path = GRIDS / f"{comid}.v{GRID_SCHEMA}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    import geopandas as gpd

    from src.data.site_view import build_site_view

    basin = _repair(gpd.GeoDataFrame.from_features(json.loads(basin_path.read_text())["features"], crs="EPSG:4326"))
    grid = build_site_view(basin, lat, lon, label=f"COMID-{comid}")
    if grid.empty:
        return None
    grid = pd.DataFrame(grid.drop(columns="geometry"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    grid.to_parquet(tmp, index=False)
    tmp.replace(path)
    return grid


def _site_data(comid: int, grid, lat: float, lon: float):
    """SiteData from cached cells -- build_virtual_site_data minus the delineation.

    crops/surplus are global-table joins on global_node_id. weather stays None (reach_row projects it onto the shared basis); basin is unread downstream.
    """
    import numpy as np

    from src.data import access

    return access.SiteData(
        site_uid=f"COMID-{comid}",
        crops=access._crops_for_grid(grid),
        grid=grid,
        surplus=access._surplus_for_grid(grid),
        basin_area=float(np.dot(grid["cell_area"].to_numpy(), grid["frac_cell_in_basin"].to_numpy())),
        sensor_location=(lon, lat),
    )


def compute_one(comid: int) -> str | None:
    """Compute and cache one reach row. None on success, else a reason ('!' prefix = permanent)."""
    path = ROWS / f"{comid}.json"
    if path.exists() and _is_current(path):
        return None
    basin_path = BASIN_CACHE / f"{comid}.json"
    if not basin_path.exists():
        return "no basin cached"  # fetch_basins has not reached it yet; retry next run

    try:
        st = _state()
        lon, lat = st["outlets"][comid]
        grid = _cells(comid, basin_path, lat, lon)
        if grid is None:
            return "!no intersecting grid cells"
        sd = _site_data(comid, grid, lat, lon)
        row = st["bf"].reach_row(comid, sd, lat, lon, st["basis"], st["tasks"], FORECAST_YEARS)
        if row is None:
            return "!no intersecting grid cells"
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(row, separators=(",", ":")))
        tmp.replace(path)
        return None
    except Exception as e:  # noqa: BLE001 -- one bad reach must not end a 3-hour pass
        return f"{type(e).__name__}: {e}"[:160]


def status() -> dict:
    ids = reach_ids()
    have = {int(p.stem) for p in ROWS.glob("*.json") if p.stem.isdigit() and _is_current(p)}
    stale = {int(p.stem) for p in ROWS.glob("*.json") if p.stem.isdigit()} - have
    basins = {int(p.stem) for p in BASIN_CACHE.glob("*.json") if p.stem.isdigit()}
    failed = json.loads(FAILED.read_text()) if FAILED.exists() else {}
    todo = [c for c in ids if c not in have and str(c) not in failed and c in basins]
    return {
        "total": len(ids),
        "basins": len(basins & set(ids)),
        "rows": len(have & set(ids)),
        "stale": len(stale & set(ids)),
        "tombstoned": len(failed),
        "todo": todo,
    }


def main(limit=None, workers=1, status_only=False) -> None:
    ROWS.mkdir(parents=True, exist_ok=True)
    st = status()
    print(
        f"reaches {st['total']:,}  basins cached {st['basins']:,}  rows built {st['rows']:,}  "
        f"tombstoned {st['tombstoned']}  ready to build {len(st['todo']):,}"
    )
    if st["stale"]:
        print(f"  ({st['stale']:,} rows are from an older reach_row shape and will be rebuilt)")
    if st["basins"] < st["total"]:
        print(f"  ({st['total'] - st['basins']:,} basins still to fetch -- run widget.static.fetch_basins)")
    if status_only:
        return
    todo = st["todo"][:limit] if limit else st["todo"]
    if not todo:
        print("nothing to build.")
        return

    print(f"building {len(todo):,} rows with {workers} worker(s). Ctrl-C is safe.\n")
    failed = json.loads(FAILED.read_text()) if FAILED.exists() else {}
    done = 0
    t0 = time.monotonic()
    # submit/as_completed, NOT map(chunksize=...). A reach costs anywhere from milliseconds to ~12 s, and map() both deals the work in fixed blocks -- so one huge basin holds up its whole block -- and yields IN ORDER, so a single slow reach freezes reporting for everything behind it while the workers race ahead. Neither is a hang, but both look exactly like one.
    #
    # No max_tasks_per_child either. Recycling a worker means re-importing the geo stack and re-reading the flowlines parquet under `spawn`, and the workers advance in lockstep so those land together -- several hundred MB allocated at once, per worker, at the same moment. The unbounded cache it was guarding against is capped at source (see src/data/d8.py).
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        if pool is not None:
            futures = {pool.submit(compute_one, c): c for c in todo}
            results = ((futures[f], f.result()) for f in as_completed(futures))
            done, failed = _drain(results, len(todo), t0, failed)
        else:
            done, failed = _drain(((c, compute_one(c)) for c in todo), len(todo), t0, failed)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {done:,}. Everything built is on disk; re-run to continue.")
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)  # do not block Ctrl-C on the running batch
        if failed:
            FAILED.write_text(json.dumps(failed, indent=1, sort_keys=True))

    end = status()
    print(f"\nrows {end['rows']:,}/{end['total']:,}  tombstoned {end['tombstoned']}  remaining {len(end['todo']):,}")


def _drain(results, total, t0, failed):
    done = 0
    for comid, reason in results:
        done += 1
        if reason and reason.startswith("!"):
            failed[str(comid)] = reason
        elif reason:
            print(f"  [retry later] {comid}: {reason}")
        if done % 200 == 0 or done == total:
            rate = done / max(time.monotonic() - t0, 1e-9)
            print(
                f"  {done:,}/{total:,}  {rate:.1f}/s  ~{(total - done) / max(rate, 1e-9) / 3600:.1f} h left", flush=True
            )
    return done, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="build at most this many, then stop")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (CPU-bound; try 6)")
    p.add_argument("--status", action="store_true", dest="status_only", help="report progress and exit")
    a = p.parse_args()
    main(limit=a.limit, workers=a.workers, status_only=a.status_only)
