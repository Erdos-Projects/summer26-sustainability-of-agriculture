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
SOURCE_DIR = Path(__file__).resolve().parents[1] / "IWQIS_archive"

SITES_DIR.mkdir(parents=True, exist_ok=True)

# files ---------------------------------------------------------------
MEASURES_SOURCE_FILE = SOURCE_DIR / "measures.csv"
METADATA_SOURCE_FILE = SOURCE_DIR / "site_clean.csv"
PARAMS_SOURCE_FILE = SOURCE_DIR / "params.csv"
MEASURES_TARGET_FILE = THIS_DIR / "iwqis_measures.csv"
METADATA_TARGET_FILE = THIS_DIR / "iwqis_site_metadata.csv"
PARAMS_TARGET_FILE = THIS_DIR / "iwqis_params.csv"


def get_site_metadata():
    return pd.read_csv(METADATA_SOURCE_FILE, engine="python", on_bad_lines="warn")


def get_full_data():
    manifest_path = SOURCE_DIR / "chunks/iwqis_alldata_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    chunks_dir = Path(manifest_path).parent
    print(f"Reassembling '{manifest['original_filename']}'")
    print(f"Expected: {manifest['total_rows']:,} rows across {manifest['num_chunks']} chunks")
    chunks_info = manifest["chunks"]
    files = [chunks_dir / Path(chunk["filename"]) for chunk in chunks_info]
    print(f"Found {len(files)} chunks to merge. Plan is to merge the files")
    for f in files:
        print(f"  {f}")
    print("in that order. Proceeding:")

    dfs = []
    for i, f in enumerate(files):
        print(f"  Reading chunk {i+1}/{len(files)}: {f}")
        dfs.append(pd.read_csv(f))

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


def og_filter_sites(sparsity_cutoff, lifespan_cutoff, source):
    # get the unique uids and store them
    uids = list(source["site_uid"].unique())

    # build dictionary of dataframes indexed by site ------------
    data_by_site = {}
    for i, uid in enumerate(uids):
        data_by_site[uid] = source[source.site_uid == uid]
        if i % 10 == 0:
            print(f"processed {i+10} / {len(uids)} uids")

    # two calculator methods ------------------------------------
    def get_year_diff(uid):
        dt = pd.to_datetime(data_by_site[uid]["datetime"], utc=True)
        early = dt.min()
        late = dt.max()
        return (late - early).total_seconds() / (365.25 * 24 * 3600)

    def measure_nitrate_data(uid):
        df = data_by_site[uid]
        num_entries = df["nitrate_con"].count()
        return num_entries / df.shape[0]

    # store the starting number of sites
    og_length = len(source.site_uid.unique())

    print(f"Calculating site quality for {og_length} sites, this may take a while...", end="", flush=True)
    # make a list of pairs (site_uid, % nonempty nitrogen data)
    vals = pd.DataFrame(
        [(uid, measure_nitrate_data(uid), get_year_diff(uid)) for uid in uids],
        columns=["site_uid", "data_count", "lifespan"],
    )
    print("done.")

    keep = vals[(vals.data_count >= sparsity_cutoff) & (vals.lifespan >= lifespan_cutoff)]
    remove = vals[(vals.data_count < sparsity_cutoff) | (vals.lifespan < lifespan_cutoff)]

    # the combined sites
    comb_length = keep.shape[0] + remove.shape[0]
    print(f"{keep.shape[0]} (keep) + {remove.shape[0]} (remove) = {keep.shape[0] + remove.shape[0]}")
    print("-----------------------------------")

    # should be equal if logic worked correctly
    assert comb_length == og_length

    return keep, remove


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


def precheck():
    """True iff the three top-level outputs exist AND the site files in
    IWQIS-sites exactly match the uids in iwqis-site-metadata."""
    targets = [PARAMS_TARGET_FILE, MEASURES_TARGET_FILE, METADATA_TARGET_FILE]
    if not all(t.exists() for t in targets):
        return False

    # uids listed in the metadata file
    meta = pd.read_csv(METADATA_TARGET_FILE)
    expected_uids = {str(u).strip() for u in meta["uid"]}

    # uids present as site files
    written_uids = {f.name[: -len("_all_data.csv")] for f in SITES_DIR.glob("*_all_data.csv")}

    return expected_uids == written_uids


def main(api_keys):

    SPARSITY_CUTOFF = 0.5
    LIFESPAN_CUTOFF = 3.92
    KNOWN_BAD = [
        "WQS0113",
        "WQS9901",
        "WQS9903",
        "WQS9904",
        "WQS9902",
        "WQS0091",
        "WQS0088",
        "WQS0090",
        "WQS0045",
        "WQS0075",
    ]  # known garbage sites that aren't picked up in filtering

    print("---- Building IWQIS Water Data ----")

    if precheck():
        print(
            f"Data already exists, skipping. To rerun, delete/rename one of\n  {METADATA_TARGET_FILE}\n  {PARAMS_TARGET_FILE}\n  {MEASURES_TARGET_FILE}"
        )
        return None

    # get the full data
    full_data = get_full_data()

    # get the filtered sites
    print("Filtering out garbage sites (this may take a minute)")
    keep, _ = filter_sites(SPARSITY_CUTOFF, LIFESPAN_CUTOFF, source=full_data)

    # filter the known bad sites, get the uids of the keepers
    keep = keep[keep.site_uid.isin(KNOWN_BAD) == False]
    keep_uids = keep.site_uid.unique()
    print(f"Checked against known garbage sites, ended with {keep_uids.shape[0]} sites.")

    # store the full datasets of the good sites
    for uid in keep_uids:
        file = SITES_DIR / f"{str(uid).strip()}_all_data.csv"
        print(f"  Saving {file}...", end="", flush=True)
        full_data[full_data.site_uid == uid].to_csv(file, index=False)
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
