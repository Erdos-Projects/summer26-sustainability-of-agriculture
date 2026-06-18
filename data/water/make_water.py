from pathlib import Path
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

THIS_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = THIS_DIR / "config" / "pipeline_config.toml"
_STATS_FILE = THIS_DIR / "metadata" / "site_statistics.csv"


def summarize_state():
    USGS_METADATA = THIS_DIR / "metadata" / "usgs_site_metadata.csv"
    IWQIS_METADATA = THIS_DIR / "metadata" / "iwqis_site_metadata.csv"
    LOC_METADATA = THIS_DIR / "metadata" / "site_location_metadata.csv"

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
        print(f"USGS sites not in combined: {list(usgs_out)}")
        print(f"IWQIS sites not in combined: {list(iwqis_out)}")

    USGS_COUNT = sum(1 for _ in Path(THIS_DIR / "sites").glob("USGS-*.parquet"))
    IWQIS_COUNT = sum(1 for _ in Path(THIS_DIR / "sites").glob("WQ*.parquet"))
    print(f"\nThere are {USGS_COUNT} USGS files in {THIS_DIR / 'sites'}.")
    print(f"There are {IWQIS_COUNT} IWQIS files in {THIS_DIR / 'sites'}.")

    BASINS_DIR = THIS_DIR.parent / "basins" / "basins1"
    USGS_BASIN_COUNT = sum(1 for _ in BASINS_DIR.glob("USGS-*.parquet"))
    IWQIS_BASIN_COUNT = sum(1 for _ in BASINS_DIR.glob("WQ*.parquet"))
    print(f"\nThere are {USGS_BASIN_COUNT} USGS basin files in {BASINS_DIR}.")
    print(f"There are {IWQIS_BASIN_COUNT} IWQIS basin files in {BASINS_DIR}.")


def gen_statistics():
    from access import get_full_data
    df = get_full_data()
    grouped = df.groupby("site_uid", sort=False)
    sparsity = grouped["nitrate_con"].count() / grouped.size()
    first_date = grouped["datetime"].min()
    last_date = grouped["datetime"].max()
    lifespan = (last_date - first_date).dt.total_seconds() / (365.25 * 24 * 3600)
    return pd.DataFrame({
        "nitrate_sparsity": sparsity,
        "start_date": first_date,
        "last_date": last_date,
        "lifespan": lifespan,
    }).reset_index()


def get_api_keys():
    import tomllib
    with open(THIS_DIR.parent / "api-keys.toml", "rb") as f:
        return tomllib.load(f)


def main(api_keys=None, force: bool = False):
    import tomllib
    import make_iwqis_data as make_iwqis
    import make_usgs_data as make_usgs
    from make_site_locations import create_site_locations

    if api_keys is None:
        api_keys = get_api_keys()

    with open(_CONFIG_FILE, "rb") as f:
        cfg = tomllib.load(f)

    all_filtered = cfg["site_filters"]["known_bad"] + cfg["site_filters"]["big_basin"]
    usgs_filter  = [uid for uid in all_filtered if uid.startswith("USGS")]
    iwqis_filter = [uid for uid in all_filtered if uid.startswith("WQ")]

    make_usgs.main(api_keys, extra_filter=usgs_filter)
    make_iwqis.main(api_keys, extra_filter=iwqis_filter)
    create_site_locations()

    stats = gen_statistics()
    stats.to_csv(_STATS_FILE, index=False)
    print(f"Saved site statistics to {_STATS_FILE}")

    summarize_state()


if __name__ == "__main__":
    main()
