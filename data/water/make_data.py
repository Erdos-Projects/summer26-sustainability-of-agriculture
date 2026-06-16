import importlib.util
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent
MAKE_USGS = DATA_DIR / "make_usgs_data.py"
MAKE_IWQIS = DATA_DIR / "make_iwqis_data.py"
MAKE_BASINS = DATA_DIR / "make_basins.py"

USGS_METADATA = DATA_DIR / "usgs_site_metadata.csv"
IWQIS_KEEPERS = DATA_DIR / "iwqis_site_metadata.csv"
SITE_CLEAN = DATA_DIR.parent / "IWQIS_archive/site_clean.csv"

SITE_LOCATION_METADATA = DATA_DIR / "site_location_metadata.csv"


def load_and_run(script, api_keys):
    # import the script properly as a module -------------------------------------------------
    spec = importlib.util.spec_from_file_location(f"dataset_{script.parent.name}", script)
    module = importlib.util.module_from_spec(spec)

    # change directory to presrve relative-path assumptions in the script --------------------
    cwd = os.getcwd()
    os.chdir(script.parent)

    # run the script -------------------------------------------------------------------------
    try:
        spec.loader.exec_module(module)  # run top-level code like global var definitions
        module.main(api_keys)  # execute the method main()
    finally:
        os.chdir(cwd)


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
    new_df.to_csv(SITE_LOCATION_METADATA, index=False)


def rename_columns():
    usgs_to_iwqis_full = {
        "time": "datetime",
        "site_id": "site_uid",
        "temp_water": "temp_water",
        "nitrate_con": "nitrate_con",
        "diss_oxy_con": "diss_oxy_con",
        "diss_oxy_sat": "diss_oxy_sat",
        "ph": "ph",
        "spec_cond": "spec_cond",
        # USGS-only columns (discharge, stage) have no IWQIS name
    }

    print("Renaming USGS columns to align with IWQIS")
    usgs_uids = pd.read_csv(USGS_METADATA).monitoring_location_id.unique()
    for uid in usgs_uids:
        name = DATA_DIR / f"sites/{uid}_all_data.csv"
        if name.exists() == False:
            print(f"Tried to rename {name} but it does not exist.")
            continue
        try:
            usgs_df = pd.read_csv(name, index_col="time")
        except ValueError:
            print(f"Columns of {name} have already been renamed.")
            continue
        usgs_df.index.name = "datetime"
        usgs_df = usgs_df.rename(columns=usgs_to_iwqis_full)
        usgs_df.to_csv(DATA_DIR / f"sites/{uid}_all_data.csv", index=True, index_label="datetime")

    print("\nRenaming concluded.")


def main(api_keys):
    def lil_helper(script):
        if not script.exists():
            raise FileNotFoundError(f"Expected {script}")

        print(f"=== {script.name} ===")
        load_and_run(script, api_keys=api_keys)

    # lil_helper(MAKE_USGS)
    # lil_helper(MAKE_IWQIS)
    create_site_locations()
    rename_columns()


if __name__ == "__main__":
    main()
