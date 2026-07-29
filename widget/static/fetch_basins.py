"""Fetch and cache the NLDI upstream basin for every forecastable NHD reach.

A dropped pin on the static site snaps to one of these reaches, and its precomputed feature row is what the light models score. The basin polygon is the input to all of it -- cells, frac_cell_in_basin, dist_to_sensor -- and NLDI is the only step in the whole pin-to-forecast chain that leaves the machine (`api.water.usgs.gov/nldi/linked-data/comid/{comid}/basin`). This script pays that cost once, offline, so the published site never has to.

RUN IT YOURSELF, AT YOUR OWN PACE. There are ~16,800 reaches at stream order >= 3, and NLDI meters
access with a QUOTA rather than a rate: `x-ratelimit-limit: 800` per window, then 429s carrying
`retry-after: ~699` until it resets. Pacing does not avoid that -- 800 requests spends the window
whether they take 13 minutes or 3 -- so the script sleeps out each reset and keeps going. Expect
the run to alternate between bursts and ~12-minute waits, and to take the best part of a day.

It is built to be interrupted:

    python -m widget.static.fetch_basins --status      # how far along, no requests
    python -m widget.static.fetch_basins               # fetch everything still missing
    python -m widget.static.fetch_basins --limit 500   # one bounded batch, then stop
    python -m widget.static.fetch_basins --workers 4   # more parallelism (mind the rate limit)

Ctrl-C is safe at any point: each response is written to its own file the moment it arrives, and a restart skips whatever is already on disk. Nothing is ever rewritten, so re-running costs only the reaches you have not got yet.

Reaches that NLDI genuinely cannot answer for (404, or an empty polygon) are recorded as tombstones in `failed.json` and skipped on later runs, so a handful of dead COMIDs cannot stall a pass forever. Delete that file to retry them. Transport errors are NOT tombstoned -- they are retried on the next run.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent  # widget/static
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

# Where the polygons land. One file per COMID: resumability is just "does this path exist".
CACHE = _ROOT / "src" / "data" / "raw" / "basins" / "nldi"
FAILED = CACHE / "failed.json"

# Reach selection. Order >= 3 is the forecastable set agreed for the static build: 16,762 of the
# 61,417 reaches with a real drainage area, median basin 68 km2. Below that the basins are smaller
# than anything the models were trained on, and a pin there snaps up to the nearest order-3 stream.
#
# Imported from build_bundle so the snap index, the reach store and this fetcher cannot describe
# different sets -- a pin that snaps to a COMID with no feature row is the failure that causes.
from widget.static.build_bundle import MIN_STREAM_ORDER  # noqa: E402

_NLDI = "https://api.water.usgs.gov/nldi/linked-data"

# NLDI enforces a QUOTA, not a rate: it answers `x-ratelimit-limit: 800` with an
# `x-ratelimit-remaining` countdown, and once that hits zero every request is a 429 carrying
# `retry-after` in seconds (observed: 699, i.e. ~12 minutes). Pacing alone does not avoid it --
# a steady 1 req/s still spends the 800 and stops dead.
#
# So the only thing that works is to WAIT OUT the reset when the server says to. Exponential
# backoff cannot: 5+10+20+40s exhausts its retries long before a ~700s window reopens, and then
# every remaining reach fails in turn without ever sleeping long enough to recover. That is
# precisely how a run stalls at exactly 800 files while still appearing to be busy.
DEFAULT_RATE = 0.4  # seconds between requests; the quota dominates, this just avoids bursting
MAX_RETRIES = 4
QUOTA_HEADER = "x-ratelimit-remaining"
RETRY_AFTER_CAP = 3600  # refuse to sleep longer than this on a single 429, however odd the header


def tombstoned() -> set[int]:
    """COMIDs NLDI has no basin for (404, or an empty polygon), from failed.json.

    Excluded from reach_ids AND from the snap index (build_forecast.build_snap_index): no basin means no feature row, so a pin able to land on one would meet "no precomputed row" instead of a forecast. Delete failed.json to retry them.
    """
    return {int(c) for c in _load_failed()}


def reach_ids(min_order: int = MIN_STREAM_ORDER) -> list[int]:
    """COMIDs of the forecastable reaches, ascending.

    TotDASqKM > 0 drops the 145 NHDPlus divergence artifacts, matching what _make_basins.snap_comid considers snappable; tombstoned reaches drop out too -- together that keeps this set, the snap index and the reach store describing the same reaches, so every reach a pin can snap to has a row.
    """
    import pandas as pd

    # pandas, not geopandas: reading an attribute-only projection of a geoparquet raises there, and
    # picking the reach set needs no geometry at all.
    fl = pd.read_parquet(
        _ROOT / "src/data/processed/map_overlays/iowa_flowlines.parquet",
        columns=["COMID", "TotDASqKM", "StreamOrde"],
    )
    keep = fl[(fl["StreamOrde"] >= min_order) & (fl["TotDASqKM"] > 0)]
    dead = tombstoned()
    return sorted(int(c) for c in keep["COMID"] if int(c) not in dead)


def _load_failed() -> dict:
    return json.loads(FAILED.read_text()) if FAILED.exists() else {}


def _quota_wait(seconds: int) -> None:
    """Sleep out an NLDI quota window, saying so.

    Announced rather than silent: a run that goes quiet for twelve minutes looks indistinguishable
    from a hang, and the first version of this script genuinely did hang here.
    """
    mins = seconds / 60
    print(f"  [quota] NLDI limit reached; waiting {seconds}s (~{mins:.0f} min) for the window to reset", flush=True)
    time.sleep(seconds + 2)  # a beat past the header, so the retry does not land on the boundary


def fetch_one(comid: int, timeout: int = 60) -> str | None:
    """Fetch one basin into the cache. Returns None on success, else a reason string.

    A reason that starts with "!" is permanent (tombstoned); anything else is transport trouble and
    will be retried on the next run.
    """
    path = CACHE / f"{comid}.json"
    if path.exists():
        return None

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = requests.get(
                f"{_NLDI}/comid/{comid}/basin", params={"f": "json", "simplified": "true"}, timeout=timeout
            )
            if r.status_code == 404:
                return "!404"
            if r.status_code == 429:
                # Quota exhausted. Sleep exactly as long as the server asks and retry the SAME
                # reach without spending an attempt -- being rate limited is not a failure of this
                # request, and counting it as one is what makes a run give up on everything.
                wait = min(int(r.headers.get("retry-after", 60) or 60), RETRY_AFTER_CAP)
                _quota_wait(wait)
                continue
            if r.status_code >= 500:
                attempt += 1
                time.sleep(2 ** attempt * 5)
                continue
            r.raise_for_status()
            payload = r.json()
            if not payload.get("features"):
                return "!empty"
            # Write via a temp file then rename: a Ctrl-C mid-write must not leave a truncated
            # file that a later run would mistake for a completed fetch.
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(payload, separators=(",", ":")))
            tmp.replace(path)
            # Spend the rest of the quota window smoothly rather than sprinting into the next 429.
            remaining = r.headers.get(QUOTA_HEADER)
            if remaining is not None and remaining.isdigit() and int(remaining) == 0:
                _quota_wait(min(int(r.headers.get("retry-after", 60) or 60), RETRY_AFTER_CAP))
            return None
        except Exception as e:  # noqa: BLE001 -- transport, JSON, anything: retry then report
            attempt += 1
            if attempt >= MAX_RETRIES:
                return f"{type(e).__name__}: {e}"[:120]
            time.sleep(2 ** attempt * 5)
    return "retries exhausted"


def status(min_order: int = MIN_STREAM_ORDER) -> dict:
    ids = reach_ids(min_order)
    have = {int(p.stem) for p in CACHE.glob("*.json") if p.stem.isdigit()}
    failed = _load_failed()
    todo = [c for c in ids if c not in have and str(c) not in failed]
    return {"total": len(ids), "cached": len(have & set(ids)), "tombstoned": len(failed), "todo": len(todo), "ids": todo}


def main(limit=None, workers=1, rate=DEFAULT_RATE, min_order=MIN_STREAM_ORDER, status_only=False) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    st = status(min_order)
    print(
        f"reaches (order >= {min_order}): {st['total']:,}  cached {st['cached']:,}  "
        f"tombstoned {st['tombstoned']}  remaining {st['todo']:,}"
    )
    if status_only:
        return
    todo = st["ids"][:limit] if limit else st["ids"]
    if not todo:
        print("nothing to fetch.")
        return

    eta_h = len(todo) * rate / workers / 3600
    print(f"fetching {len(todo):,} with {workers} worker(s) at {rate}s each -- about {eta_h:.1f} h. Ctrl-C is safe.\n")

    failed = _load_failed()
    done = 0
    t0 = time.monotonic()

    def work(comid):
        time.sleep(rate)  # crude per-worker pacing; keeps the aggregate under the NLDI limit
        return comid, fetch_one(comid)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for comid, reason in pool.map(work, todo):
                done += 1
                if reason and reason.startswith("!"):
                    failed[str(comid)] = reason
                elif reason:
                    print(f"  [retry later] {comid}: {reason}")
                if done % 100 == 0 or done == len(todo):
                    rate_s = done / max(time.monotonic() - t0, 1e-9)
                    left = (len(todo) - done) / max(rate_s, 1e-9) / 3600
                    print(f"  {done:,}/{len(todo):,}  {rate_s:.2f}/s  ~{left:.1f} h left", flush=True)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {done:,}. Everything fetched is on disk; re-run to continue.")
    finally:
        if failed:
            FAILED.write_text(json.dumps(failed, indent=1, sort_keys=True))

    end = status(min_order)
    print(f"\ncached {end['cached']:,}/{end['total']:,}  tombstoned {end['tombstoned']}  remaining {end['todo']:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="fetch at most this many, then stop")
    p.add_argument("--workers", type=int, default=1, help="parallel requests (default 1)")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE, help="seconds between requests per worker")
    p.add_argument("--min-order", type=int, default=MIN_STREAM_ORDER, help="minimum NHD stream order")
    p.add_argument("--status", action="store_true", dest="status_only", help="report progress and exit")
    a = p.parse_args()
    main(limit=a.limit, workers=a.workers, rate=a.rate, min_order=a.min_order, status_only=a.status_only)
