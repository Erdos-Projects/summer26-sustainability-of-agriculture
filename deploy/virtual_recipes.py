"""Feature frame for a virtual (ungauged) SiteData -- no target, ready to score.

Reuses recipes3.build_feature_frame (the exact assembly the gauged recipes use) with a
weather-derived spine = the TARGET_YEAR daily dates. The SiteData.weather spans TARGET_YEAR
+/- 2 months, so the trailing rolling/lag features have lead-in; trimming the spine to the
year keeps that buffer for lookback while emitting only in-year rows.
"""

import sys
import importlib.util
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent  # deploy/
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))  # for the sibling build_virtual_basin module


def _load_recipes3():
    """Import isaac/recipes3.py WITHOUT adding all of isaac/ to sys.path (which would shadow
    this package's build_virtual_basin with isaac's superseded draft of the same name)."""
    if "recipes3" in sys.modules:
        return sys.modules["recipes3"]
    spec = importlib.util.spec_from_file_location("recipes3", _ROOT / "isaac" / "recipes3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recipes3"] = mod
    spec.loader.exec_module(mod)
    return mod


recipes3 = _load_recipes3()

from build_virtual_basin import TARGET_YEAR


def target_year_spine(site_data, target_year: int = TARGET_YEAR) -> pd.DatetimeIndex:
    """Daily DatetimeIndex over `target_year`, taken from the site's weather dates (which span
    target_year +/- 2 months). Trimming to the year drops the lookback buffer from the OUTPUT
    rows while keeping it available to the rolling/lag feature computations."""
    wd = pd.DatetimeIndex(sorted(pd.to_datetime(site_data.weather["date"].unique())))
    ystart = pd.Timestamp(f"{target_year}-01-01")
    yend = pd.Timestamp(f"{target_year}-12-31")
    return wd[(wd >= ystart) & (wd <= yend)]


def virtual_recipe(site_data, task: str = "reg", target_year: int = TARGET_YEAR) -> pd.DataFrame:
    """Feature frame (no target) for a virtual SiteData, task 'reg' or 'clf'.

    Columns match _best_features_REG / _best_features_CLF (i.e. recipe_REG/_CLF minus the
    target). Align to a trained model's feature_names before scoring (see deploy.predict).
    """
    spine = target_year_spine(site_data, target_year)
    if len(spine) == 0:
        raise ValueError(
            f"No weather dates fall in {target_year}; the SiteData weather window does not cover it."
        )
    return recipes3.build_feature_frame(site_data, task=task, spine=spine)
