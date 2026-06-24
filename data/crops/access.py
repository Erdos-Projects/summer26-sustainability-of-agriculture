"""Read-only access layer for crop data in data/crops/."""

import sys
import pandas as pd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "crops_data"
_GRID_AGG_DIR = _DATA_DIR / "grid"  # per-site crops aggregated onto the rain grid
_META_DIR = _THIS_DIR / "crops_meta"
_RAW_DIR = _THIS_DIR / "crops_raw"

sys.path.insert(0, _THIS_DIR.parents[1])
from data.crops.cdl_legend import cdl_legend


def lookup_crop(index):
    """Return the English name of the crop denoted by CDL code <index>"""
    return cdl_legend[index]


def get_crops(site_uid: str) -> pd.DataFrame:
    """Return a site's CDL crops aggregated onto the rain grid.

    Columns: node_id, global_node_id, year, then one pixel-count column per crop
    class produced by the remap in make_crops.py (e.g. Corn, Soybeans, ...,
    Other). Join to the rain grid (data.get_rain_grid) on node_id for
    coordinates; global_node_id is the canonical IEM cell index, shared across
    basins.

    Parameters
    ----------
    site_uid : str
        The site unique identifier.

    Returns
    -------
    pandas.DataFrame
        The per-(node_id, year) crop counts.

    Raises
    ------
    FileNotFoundError
        If the parquet hasn't been generated yet (run make_crops.py).
    """
    path = _GRID_AGG_DIR / f"{site_uid}_crops_grid.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No crops data for {site_uid}. Run make_crops.py to generate it.")
    return pd.read_parquet(path)
