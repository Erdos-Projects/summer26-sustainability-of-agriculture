"""Build site_location_metadata.csv from USGS and IWQIS metadata.

Combines USGS site coordinates (authoritative) with the filtered IWQIS site
list to produce a single location table used by the access layer.
"""

from pathlib import Path
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
USGS_METADATA = THIS_DIR / "metadata" / "usgs_site_metadata.csv"
IWQIS_KEEPERS = THIS_DIR / "metadata" / "iwqis_site_metadata.csv"
SITE_CLEAN = THIS_DIR.parent / "archive" / "IWQIS_archive" / "site_clean.csv"
SITE_LOCATION_METADATA = THIS_DIR / "metadata" / "site_location_metadata.csv"


def create_site_locations():
    usgs_metadata = pd.read_csv(USGS_METADATA)
    site_clean = pd.read_csv(SITE_CLEAN)

    usgs_metadata = usgs_metadata.rename(
        columns={"state_name": "state", "monitoring_location_id": "uid"}
    )

    coords = usgs_metadata["geometry"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    usgs_metadata["longitude"] = coords[0]
    usgs_metadata["latitude"] = coords[1]
    usgs_metadata = usgs_metadata.drop(columns=["geometry"])

    uid_overlap = [
        uid for uid in usgs_metadata.uid.unique()
        if (site_clean.uid == str(uid).replace("USGS-", "")).any()
    ]
    uid_diff = set(usgs_metadata.uid.unique()).difference(set(uid_overlap))

    for uid in uid_overlap:
        site_clean.loc[site_clean.uid == str(uid).replace("USGS-", ""), "uid"] = uid
        lat = float(usgs_metadata.loc[usgs_metadata.uid == uid, "latitude"].iloc[0])
        lon = float(usgs_metadata.loc[usgs_metadata.uid == uid, "longitude"].iloc[0])
        site_clean.loc[site_clean.uid == uid, "latitude"] = lat
        site_clean.loc[site_clean.uid == uid, "longitude"] = lon

    keeper_ids = pd.read_csv(IWQIS_KEEPERS).uid.unique().tolist()
    site_clean = site_clean[site_clean.uid.isin(keeper_ids + uid_overlap)]

    new_rows = usgs_metadata[usgs_metadata.uid.isin(uid_diff)][
        ["state", "uid", "latitude", "longitude"]
    ].drop_duplicates()

    new_df = pd.concat([site_clean, new_rows], ignore_index=True)
    new_df = new_df.rename(columns={"uid": "site_uid"})
    new_df.to_csv(SITE_LOCATION_METADATA, index=False)


if __name__ == "__main__":
    create_site_locations()
    print(f"Wrote {SITE_LOCATION_METADATA}")
