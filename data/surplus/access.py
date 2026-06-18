"""Read-only access layer for per-site nitrogen surplus data."""

import pandas as pd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "surplus_data"


def get_surplus(site_uid: str) -> pd.DataFrame:
    """Return the surplus DataFrame for a single site.

    Columns: year, surplus_kgha, total_kg_N, x, y, lon, lat

    Raises FileNotFoundError if the parquet hasn't been generated yet.
    Run make_surplus.py to build it.
    """
    path = _DATA_DIR / f"{site_uid}_surplus.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No surplus data for {site_uid}. Run make_surplus.py to generate it."
        )
    return pd.read_parquet(path)
