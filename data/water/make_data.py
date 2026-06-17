from pathlib import Path
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_iwqis_data as make_iwqis
import make_usgs_data as make_usgs
import make_basins as make_basins

DATA_DIR = Path(__file__).resolve().parent
MAKE_USGS = DATA_DIR / "make_usgs_data.py"
MAKE_IWQIS = DATA_DIR / "make_iwqis_data.py"
MAKE_BASINS = DATA_DIR / "make_basins.py"

USGS_METADATA = DATA_DIR / "usgs_site_metadata.csv"
IWQIS_KEEPERS = DATA_DIR / "iwqis_site_metadata.csv"
SITE_CLEAN = DATA_DIR.parent / "IWQIS_archive/site_clean.csv"

SITE_LOCATION_METADATA = DATA_DIR / "site_location_metadata.csv"
SENTINEL = DATA_DIR / ".pipeline_complete"


def create_site_locations():
    # keep only filtered keeper WQ sites or USGS sites
    usgs_metadata = pd.read_csv(USGS_METADATA)
    site_clean = pd.read_csv(SITE_CLEAN)

    # modify the usgs_metadata column names
    usgs_metadata = usgs_metadata.rename(columns={"state_name": "state", "monitoring_location_id": "uid"})

    # convert geoemetry to latitude and longitude, delete
    coords = usgs_metadata["geometry"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    usgs_metadata["longitude"] = coords[0]
    usgs_metadata["latitude"] = coords[1]
    usgs_metadata = usgs_metadata.drop(columns=["geometry"])

    # get uids in and outside the site_clean metadata
    uid_overlap = [uid for uid in usgs_metadata.uid.unique() if (site_clean.uid == str(uid).replace("USGS-", "")).any()]
    uid_diff = set(usgs_metadata.uid.unique()).difference(set(uid_overlap))

    # replace the uids, latitudes and longitudes in site_clean \cap usgs_metadata
    for uid in uid_overlap:
        site_clean.loc[site_clean.uid == str(uid).replace("USGS-", ""), "uid"] = uid

        lat1 = float(usgs_metadata.loc[usgs_metadata.uid == uid, "latitude"].iloc[0])
        lon1 = float(usgs_metadata.loc[usgs_metadata.uid == uid, "longitude"].iloc[0])
        lat2 = float(site_clean.loc[site_clean.uid == uid, "latitude"].iloc[0])
        lon2 = float(site_clean.loc[site_clean.uid == uid, "longitude"].iloc[0])

        site_clean.loc[site_clean.uid == uid, "latitude"] = lat1
        site_clean.loc[site_clean.uid == uid, "longitude"] = lon1

    # remove the bad sites from site_clean
    keeper_ids = pd.read_csv(IWQIS_KEEPERS).uid.unique().tolist()
    site_clean = site_clean[site_clean.uid.isin(keeper_ids + uid_overlap)]

    # get the final rows
    new_rows = usgs_metadata[usgs_metadata.uid.isin(uid_diff)][
        ["state", "uid", "latitude", "longitude"]
    ].drop_duplicates()

    # create the new locations file
    new_df = pd.concat([site_clean, new_rows], ignore_index=True)
    new_df = new_df.rename(columns={"uid": "site_uid"})
    new_df.to_csv(SITE_LOCATION_METADATA, index=False)


def summarize_state():
    """Basic summary of the state of the data"""
    USGS_METADATA = DATA_DIR / "usgs_site_metadata.csv"
    IWQIS_METADATA = DATA_DIR / "iwqis_site_metadata.csv"
    LOC_METADATA = DATA_DIR / "site_location_metadata.csv"

    unique_usgs_m = pd.read_csv(USGS_METADATA).monitoring_location_id.unique()
    unique_iwqis_m = pd.read_csv(IWQIS_METADATA).uid.unique()
    unique_loc_m = pd.read_csv(LOC_METADATA).site_uid.unique()

    print(f"\n--- SUMMARY OF WATER -----------------")
    print(f" usgs metadata sites: {len(unique_usgs_m)}")
    print(f"iwqis metadata sites: {len(unique_iwqis_m)}")
    print(f"   combined metadata: {len(unique_loc_m)}")
    if len(unique_iwqis_m) + len(unique_usgs_m) == len(unique_loc_m):
        print(f"{len(unique_usgs_m)} (usgs) + {len(unique_iwqis_m)} (iwqis) = {len(unique_loc_m)} (combined)")
    else:
        usgs_out = set(unique_usgs_m).difference(set(unique_loc_m))
        iwqis_out = set(unique_iwqis_m).difference(set(unique_loc_m))
        print(f"!! {len(unique_iwqis_m)} (usgs) + {len(unique_usgs_m)} (iwqis) != {len(unique_loc_m)} (combined)!!")
        print(f"""The USGS sites
              \n   {list(usgs_out)}\nin
              \n   {USGS_METADATA}
              \nare not in the combined metadata""")
        print(f"""The IWQIS sites
              \n   {list(iwqis_out)}\nin
              \n   {IWQIS_METADATA}
              \nare not in the combined metadata""")

    USGS_COUNT = sum(1 for _ in Path(DATA_DIR / "sites").glob("USGS-*.parquet"))
    IWQIS_COUNT = sum(1 for _ in Path(DATA_DIR / "sites").glob("WQ*.parquet"))

    print(f"\nThere are {USGS_COUNT} USGS files in {DATA_DIR / "sites"}.")
    print(f"There are {IWQIS_COUNT} IWQIS files in {DATA_DIR / "sites"}.")

    USGS_BASIN_COUNT = sum(1 for _ in Path(DATA_DIR / "basins").glob("USGS-*.parquet"))
    IWQIS_BASIN_COUNT = sum(1 for _ in Path(DATA_DIR / "basins").glob("WQ*.parquet"))

    print(f"\nThere are {USGS_BASIN_COUNT} USGS basin files in {DATA_DIR / "basins"}.")
    print(f"There are {IWQIS_BASIN_COUNT} IWQIS basin files in {DATA_DIR / "basins"}.")


def precheck():
    """Basic precheck step to avoid rebuilding

    Returns
    -------
    bool
        True if data already built
    """
    return SENTINEL.exists()


def main(api_keys):
    if precheck():
        print(f"Water pipeline already complete. Delete {SENTINEL} to rebuild.")
        summarize_state()
        return

    make_usgs.main(api_keys)
    make_iwqis.main(api_keys)
    create_site_locations()
    make_basins.main(api_keys)

    SENTINEL.touch()


if __name__ == "__main__":
    main()
