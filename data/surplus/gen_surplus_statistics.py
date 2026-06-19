import pandas as pd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent
_PROJ_ROOT = _DATA_DIR.parent
_RAW_DIR = _THIS_DIR / "surplus_raw"
_META_DIR = _THIS_DIR / "surplus_meta"
_STATS_FILE = _META_DIR / "surplus_stats.csv"

import sys

sys.path.insert(0, str(_PROJ_ROOT))

from data import basins


def gen_surplus_statistics():
    """Generates surplus statistics, one row for each year, and saves it to file.

    Currently computes min and max of both surplus and total N.
    Throws away values of zero before computing the max and min
    """
    print(f"Creating surplus stats at {_STATS_FILE}...", flush=True, end="")
    # columns ['year', 'surplus_kgha', 'total_kg_N', 'x', 'y', 'lon', 'lat']
    surplus_df = pd.read_parquet(_RAW_DIR / "iowa_nitrogen_surplus.parquet")

    nonzero_df = surplus_df[(surplus_df.surplus_kgha != 0) & (surplus_df.total_kg_N != 0)]

    # get only the year and surplus_kgha
    grouped = nonzero_df[["year", "surplus_kgha", "total_kg_N"]].groupby("year")
    stats = grouped.agg(
        min_surplus_kgha=("surplus_kgha", "min"),
        max_surplus_kgha=("surplus_kgha", "max"),
        min_total_kg_N=("total_kg_N", "max"),
        max_total_kg_N=("total_kg_N", "max"),
    )
    stats.to_csv(_STATS_FILE)
    print("done.")
    return stats


if __name__ == "__main__":
    stats = gen_surplus_statistics()
    print(stats.head())
