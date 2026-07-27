"""Build the water dataset: nitrate target + site metadata (faithful port of make_water.py).

NOT a re-grain -- water is the prediction target (per-site nitrate time series) plus the
canonical site metadata (the 83-site list, locations, date ranges), so it stays per-site. This
orchestrates the two sources and the combiner:

    usgs   (network, dataretrieval)     -> processed/water/data/USGS-*.parquet + usgs metadata
    iwqis  (offline chunk reassembly)   -> processed/water/data/WQS*.parquet  + iwqis metadata
    site_locations (combine)            -> processed/water/meta/site_location_metadata.csv
    prune_filtered_sites (exclusions)   -> removes [site_filters] sites from the built artifacts
    gen_statistics                      -> processed/water/meta/site_statistics.csv

The per-site parquets under processed/water are the DURABLE source of truth (IWQIS is not
re-fetchable) and are tracked in git; see .gitignore. api-keys.toml (for USGS) is read from
api-keys.toml (repo root), else the legacy sustag/data/api-keys.toml.

Usage
-----
    python -m src.build._make_water                # full build (network + raw sources)
    python -m src.build._make_water --prune-only   # apply [site_filters] to the built artifacts
"""

import argparse
import shutil
import sys
import tomllib
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent  # src/build/
_SRC = _THIS_DIR.parent  # src/
sys.path.insert(0, str(_SRC.parent))  # repo root on path

from src.build.config import get_config
from src.build.util import iwqis, site_locations, usgs
from src.build.util._water_paths import data_dir, meta_dir


def get_api_keys() -> dict:
    """Load api-keys.toml. Needed for the USGS pull."""
    p = _THIS_DIR / "api-keys.toml"
    if not p.exists():
        raise FileNotFoundError(f"api-keys.toml not found at {p} (needed for the USGS pull).")
    with open(p, "rb") as f:
        return tomllib.load(f)


def filtered_uids() -> list:
    """Every site uid excluded by config [site_filters] (known_bad + big_basin + groundwater)."""
    sf = get_config()["site_filters"]
    return sf["known_bad"] + sf["big_basin"] + sf.get("groundwater", [])


# Every per-site table keyed by uid, as (path, uid column). site_location_metadata.csv is the one
# access.get_site_ids() reads, so it is the one that actually shrinks the site list; the rest are
# pruned to keep them consistent with it.
#
# usgs_site_metadata.csv is deliberately NOT here. usgs.main regenerates it over the NETWORK
# whenever its uid set stops matching USGS_SITE_LIST, so pruning it would force a re-fetch on
# every subsequent run. It is a source catalogue, not a site list.
def _keyed_tables():
    basins_meta = _SRC / "data" / "processed" / "basins" / "meta"
    return [
        (meta_dir() / "iwqis_site_metadata.csv", "uid"),
        (meta_dir() / "site_location_metadata.csv", "site_uid"),
        (meta_dir() / "site_statistics.csv", "site_uid"),
        (basins_meta / "preferred_basin.csv", "site_uid"),
        (basins_meta / ".preferred_basin_archive.csv", "site_uid"),
    ]


def prune_filtered_sites(uids=None) -> None:
    """Remove excluded sites from the BUILT artifacts (per-site parquets + uid-keyed tables).

    The fetch-time filters (usgs/iwqis extra_filter) only bite on a fresh build from raw sources.
    The IWQIS chunks are not re-fetchable and are absent here, so iwqis.main short-circuits on its
    _precheck and never rewrites its metadata; processed/ is the durable source of truth. An
    exclusion therefore has to be applied to it directly, or it is a no-op.

    Also clears src/data/cache/, which holds the cross-site nitrate climatology. That cache is
    keyed on the SITE SET (a wide frame with one column per site), so leaving it would keep an
    excluded site inside every neighbour feature -- silently, and forever.

    Idempotent: skips anything already gone, and only rewrites a table it actually changed.
    """
    uids = list(filtered_uids() if uids is None else uids)
    removed_parquets, removed_rows, removed_basins = [], {}, []

    for uid in uids:
        p = data_dir() / f"{uid}_water.parquet"
        if p.exists():
            p.unlink()
            removed_parquets.append(uid)
        for b in (_SRC / "data" / "processed" / "basins" / "data").glob(f"{uid}_basin*.parquet"):
            b.unlink()
            removed_basins.append(b.name)

    for path, key in _keyed_tables():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if key not in df.columns:
            continue
        drop = df[key].astype(str).isin(uids)
        if drop.any():
            df[~drop].to_csv(path, index=False)
            removed_rows[path.name] = sorted(df.loc[drop, key].astype(str))

    cache = _SRC / "data" / "cache"
    cleared_cache = cache.exists() and any(cache.iterdir())
    if cleared_cache:
        shutil.rmtree(cache, ignore_errors=True)

    if removed_parquets:
        print(f"Pruned {len(removed_parquets)} water parquet(s): {removed_parquets}")
    if removed_basins:
        print(f"Pruned {len(removed_basins)} basin parquet(s): {removed_basins}")
    for name, dropped in removed_rows.items():
        print(f"Pruned {len(dropped)} row(s) from {name}: {dropped}")
    if cleared_cache:
        print("Cleared src/data/cache (cross-site nitrate climatology is keyed on the site set).")
    if not (removed_parquets or removed_basins or removed_rows or cleared_cache):
        print("Nothing to prune -- no excluded site is present in the built artifacts.")


def gen_statistics() -> pd.DataFrame:
    """Per-site nitrate stats from the built per-site parquets."""
    from src.data.access import get_all_water

    df = get_all_water()
    grouped = df.groupby("site_uid", sort=False)
    sparsity = grouped["nitrate_con"].count() / grouped.size()
    first_date = grouped["datetime"].min()
    last_date = grouped["datetime"].max()
    lifespan = (last_date - first_date).dt.total_seconds() / (365.25 * 24 * 3600)
    return pd.DataFrame(
        {"nitrate_sparsity": sparsity, "start_date": first_date, "last_date": last_date, "lifespan": lifespan}
    ).reset_index()


def main(api_keys=None, force: bool = False) -> None:
    if api_keys is None:
        api_keys = get_api_keys()

    all_filtered = filtered_uids()
    usgs_filter = [uid for uid in all_filtered if uid.startswith("USGS")]
    iwqis_filter = [uid for uid in all_filtered if uid.startswith("WQ")]

    usgs.main(api_keys, extra_filter=usgs_filter)
    iwqis.main(api_keys, extra_filter=iwqis_filter)
    site_locations.create_site_locations()

    # create_site_locations rebuilds the location table from the (unfiltered) raw registry, so the
    # prune runs after it and before gen_statistics -- the stats are derived from the kept set.
    prune_filtered_sites(all_filtered)

    stats = gen_statistics()
    stats_path = meta_dir() / "site_statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Saved site statistics to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Apply the config [site_filters] exclusions to the already-built artifacts and exit. "
        "No network, no raw sources needed -- the useful path in a clone that only has processed/.",
    )
    args = parser.parse_args()
    if args.prune_only:
        prune_filtered_sites()
    else:
        main()
