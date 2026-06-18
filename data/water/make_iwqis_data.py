"""Make the iwqis-data.

The original IWQIS-data was a downloaded zip file consisting of
- iwqis_alldata.csv: the full dataset
- measures.csv: metadata, note used
- params.csv: units of measure for the parameters
- site.csv: useful metadata about the sites themselves

Two modifications were made to this data.
1. iwqis_alldata.csv was chunked using `split_csv.py` to produce the data in `./chunks/` so the full data would fit on github
2. a line (row 63) in `site.csv` had bad quote escaping that messed up parsing, this was fixed by hand to produce `site_clean.csv`.

This script assumes the data is in the state following the modifications. It uses `reassemble.py` to reassemble the full dataset and then filters out the garbage sites to produce a final list of iwqis sites. The output of this process should be
- `IWQIS-sites/<site_uid>_all_data.csv`: one file per site of interest
- `iwqis-site-metadata.csv`: `site.csv` with garbage sites thrown out
- `iwqis-measures.csv`: just measures.csv renamed
- `iwqis-params.csv`: just params.csv renamed
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# directories ---------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent  # the directory in which this file is located
SITES_DIR = THIS_DIR / "sites"  # main target directory
SOURCE_DIR = Path(__file__).resolve().parents[1] / "archive" / "IWQIS_archive"

SITES_DIR.mkdir(parents=True, exist_ok=True)

# files ---------------------------------------------------------------
MEASURES_SOURCE_FILE = SOURCE_DIR / "measures.csv"
METADATA_SOURCE_FILE = SOURCE_DIR / "site_clean.csv"
PARAMS_SOURCE_FILE = SOURCE_DIR / "params.csv"
MEASURES_TARGET_FILE = THIS_DIR / "metadata" / "iwqis_measures.csv"
METADATA_TARGET_FILE = THIS_DIR / "metadata" / "iwqis_site_metadata.csv"
PARAMS_TARGET_FILE = THIS_DIR / "metadata" / "iwqis_params.csv"

_CONFIG_FILE = THIS_DIR / "config" / "pipeline_config.toml"

FULL_DATA = None


# lazy loading for the full dataset
def _full_data():
    global FULL_DATA
    if FULL_DATA is None:
        FULL_DATA = _get_full_data()
    return FULL_DATA


def get_site_metadata():
    return pd.read_csv(METADATA_SOURCE_FILE, engine="python", on_bad_lines="warn")


def _get_full_data():
    manifest_path = SOURCE_DIR / "chunks/iwqis_alldata_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    chunks_dir = Path(manifest_path).parent
    print(f"Reassembling '{manifest['original_filename']}'")
    print(f"Expected: {manifest['total_rows']:,} rows across {manifest['num_chunks']} chunks")
    chunks_info = manifest["chunks"]
    files = [chunks_dir / Path(chunk["filename"]) for chunk in chunks_info]
    print(
        f"Found {len(files)} chunks to merge. Plan is to merge them in the order specified in the manifest file\n  {manifest_path.parents[1].name + manifest_path.parent.name + manifest_path.name}"
    )
    print("Proceeding...")

    dfs = []
    for i, f in enumerate(files):
        print(f"  Reading chunk {i+1}/{len(files)}: {f}", end="\r")
        dfs.append(pd.read_csv(f))
    print(" ")  # ensures the last printout above remains

    full_data = pd.concat(dfs, ignore_index=True)
    print(f"Dataset reassembled. Running checks...")

    actual = full_data.shape[0]
    expected = int(manifest["total_rows"])
    if actual != expected:
        raise ValueError(
            f"Row count mismatch reassembling '{manifest['original_filename']}': "
            f"expected {expected:,}, got {actual:,} "
            f"(difference of {actual - expected:+,})"
        )
    else:
        print("All checks passed.")

    print(f"Reassembled {len(full_data):,} rows to construct the full dataset.")
    return full_data


def filter_sites(sparsity_cutoff, lifespan_cutoff, source):
    # parse datetimes once for the whole column, not per-uid
    source = source.copy()
    source["datetime"] = pd.to_datetime(source["datetime"], utc=True)

    grouped = source.groupby("site_uid", sort=False)

    # fraction of rows with non-null nitrate, per site
    data_count = grouped["nitrate_con"].count() / grouped.size()

    # lifespan in years, per site
    span = grouped["datetime"].max() - grouped["datetime"].min()
    lifespan = span.dt.total_seconds() / (365.25 * 24 * 3600)

    vals = pd.DataFrame(
        {
            "data_count": data_count,
            "lifespan": lifespan,
        }
    ).reset_index()  # site_uid becomes a column

    keep_mask = (vals.data_count >= sparsity_cutoff) & (vals.lifespan >= lifespan_cutoff)
    keep = vals[keep_mask]
    remove = vals[~keep_mask]

    print(f"{keep.shape[0]} (keep) + {remove.shape[0]} (remove) = {vals.shape[0]}")
    print("-----------------------------------")
    assert keep.shape[0] + remove.shape[0] == vals.shape[0]

    return keep, remove


def _precheck(extra_filter=[]) -> bool:
    """Return True if all expected outputs already exist.

    Uses the existing iwqis_site_metadata.csv as a manifest — if it lists N sites,
    checks that all N parquets and the two ancillary metadata files are present.
    Avoids loading _full_data() entirely when everything is already built.
    """
    if not METADATA_TARGET_FILE.exists():
        return False
    keeper_uids = pd.read_csv(METADATA_TARGET_FILE)["uid"].tolist()
    keeper_uids = [u for u in keeper_uids if u not in extra_filter]
    all_parquets = all((SITES_DIR / f"{uid}_all_data.parquet").exists() for uid in keeper_uids)
    all_meta = MEASURES_TARGET_FILE.exists() and PARAMS_TARGET_FILE.exists()
    return all_parquets and all_meta


def evaluate_uids(sparsity_cutoff, lifespan_cutoff, extra_filter=[]):
    """Return list of sites whose data needs to be created"""
    # check metadata matches the data
    current_uids = {f.name[: -len("_all_data.parquet")] for f in SITES_DIR.glob("WQ*.parquet")}

    # filter the data and return any missing sites
    print("Filtering the sites, this may take a minute...", end="", flush=True)
    keep, _ = filter_sites(sparsity_cutoff, lifespan_cutoff, source=_full_data())
    keep = keep[keep.site_uid.isin(extra_filter) == False]
    keep_uids = set(keep.site_uid.unique())
    missing_uids = list(keep_uids.difference(current_uids))
    print("done.")

    return list(keep_uids), missing_uids


def main(api_keys=None, extra_filter=[]):
    """Makes the IWQIS dataset

    Parameters
    ----------
    api_keys : dict
        dictionary containing the api_keys needed for access. Not required here.
    extra_filter : list, optional
        list of extra site_uids to filter out.

    Returns
    -------
    _type_
        _description_
    """

    if _precheck(extra_filter):
        print("IWQIS data already complete, skipping.")
        return

    import tomllib

    with open(_CONFIG_FILE, "rb") as f:
        _cfg = tomllib.load(f)["iwqis"]
    SPARSITY_CUTOFF = _cfg["sparsity_cutoff"]
    LIFESPAN_CUTOFF = _cfg["lifespan_cutoff"]

    print("---- Building IWQIS Water Data ----")

    keep_uids, missing_uids = evaluate_uids(SPARSITY_CUTOFF, LIFESPAN_CUTOFF, extra_filter=extra_filter)
    if missing_uids == []:
        print(f"No data missing. Writing metadata.")

    else:
        full_data = _full_data()
        # store the full datasets of the good sites
        for uid in missing_uids:
            file = SITES_DIR / f"{str(uid).strip()}_all_data.parquet"
            print(f"  Saving {file}...", end="", flush=True)
            site_df = full_data[full_data.site_uid == uid].copy()
            site_df["datetime"] = pd.to_datetime(site_df["datetime"], utc=True)
            site_df = site_df.set_index("datetime")
            site_df.to_parquet(file)
            print("done.")

    # store the new relevant site metadata
    print(f"Saving site metadata to {METADATA_TARGET_FILE}...", end="", flush=True)
    site_df = get_site_metadata()
    site_df[site_df.uid.isin(keep_uids)].to_csv(METADATA_TARGET_FILE, index=False)
    print("done.")

    # save the params file
    print(f"Saving params data to {PARAMS_TARGET_FILE}...", end="", flush=True)
    pd.read_csv(PARAMS_SOURCE_FILE).to_csv(PARAMS_TARGET_FILE, index=False)
    print(f"done.")

    # save the measures file
    print(f"Saving measures data to {MEASURES_TARGET_FILE}...", end="", flush=True)
    pd.read_csv(MEASURES_SOURCE_FILE).to_csv(MEASURES_TARGET_FILE, index=False)
    print(f"done.")

    # sanity check
    print("\nSanity Check:")
    print(f"Expect to keep {len(keep_uids)}")
    print(f"Saved meta data on {site_df[site_df.uid.isin(keep_uids)].shape[0]} sites")
    print("\nIWQIS data build complete.")


if __name__ == "__main__":
    main(None)
