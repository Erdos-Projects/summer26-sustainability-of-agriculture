"""Compute the per-reach feature rows the static forecast scores, one file per reach.

This is the second half of the offline forecast build. widget/static/fetch_basins.py gets the basin
polygons from NLDI; this turns each into the feature row a browser needs, using the same
recipes._agg_block the training path uses so the two cannot drift.

    python -m widget.static.build_reaches --status       # progress, computes nothing
    python -m widget.static.build_reaches                # everything still missing
    python -m widget.static.build_reaches --limit 200    # one bounded batch
    python -m widget.static.build_reaches --workers 6    # parallel (this is CPU-bound)

SEPARATE FROM THE PACK STEP ON PURPOSE. The expensive part is per reach and irreversible-ish: a D8
flow-distance field plus a cell intersection, ~0.73 s mean and 4.8 s for the largest basins, so
roughly 3.4 h single-threaded over 16,762 reaches. Writing one JSON per reach here means changing
the SHIPPED format later is a re-pack (seconds) rather than another pass (hours). Same reason the
NLDI polygons are cached rather than streamed.

Ctrl-C is safe: each row is written whole via a temp file and renamed, and a restart skips what
exists. Reaches whose basin intersects no grid cell -- real, for the smallest ones -- are recorded
as tombstones rather than retried forever.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # widget/static
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from widget.static.build_bundle import FORECAST_YEARS  # noqa: E402
from widget.static.fetch_basins import CACHE as BASIN_CACHE, reach_ids  # noqa: E402

ROWS = _ROOT / "src" / "data" / "cache" / "reach_rows"
FAILED = ROWS / "failed.json"

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


def compute_one(comid: int) -> str | None:
    """Compute and cache one reach row. None on success, else a reason ('!' prefix = permanent)."""
    path = ROWS / f"{comid}.json"
    if path.exists() and _is_current(path):
        return None
    basin_path = BASIN_CACHE / f"{comid}.json"
    if not basin_path.exists():
        return "no basin cached"  # fetch_basins has not reached it yet; retry next run

    try:
        import geopandas as gpd

        st = _state()
        lon, lat = st["outlets"][comid]
        basin = gpd.GeoDataFrame.from_features(json.loads(basin_path.read_text())["features"], crs="EPSG:4326")
        row = st["bf"].reach_row(comid, basin, lat, lon, st["basis"], st["tasks"], FORECAST_YEARS)
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
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = zip(todo, pool.map(compute_one, todo, chunksize=8))
                done, failed = _drain(results, len(todo), t0, failed)
        else:
            done, failed = _drain(((c, compute_one(c)) for c in todo), len(todo), t0, failed)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {done:,}. Everything built is on disk; re-run to continue.")
    finally:
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
            print(f"  {done:,}/{total:,}  {rate:.1f}/s  ~{(total - done) / max(rate, 1e-9) / 3600:.1f} h left", flush=True)
    return done, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="build at most this many, then stop")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (CPU-bound; try 6)")
    p.add_argument("--status", action="store_true", dest="status_only", help="report progress and exit")
    a = p.parse_args()
    main(limit=a.limit, workers=a.workers, status_only=a.status_only)
