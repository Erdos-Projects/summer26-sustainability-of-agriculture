"""Read-only access layer for crop data in data/crops/."""

import pandas as pd
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "crops_data"
_META_DIR = _THIS_DIR / "crops_meta"
_RAW_DIR = _THIS_DIR / "crops_raw"


def get_crops(site_uid: str):
    """Get the crop data corresponding to a specific site.

    Not implemented, waiting on data.

    Parameters
    ----------
    site_uid : str
        The site unique identifier

    Returns
    -------
    pandas.DataFrame
        the crop data
    """
    return pd.DataFrame([])
