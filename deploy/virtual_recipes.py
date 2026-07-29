"""Feature frame for a virtual (ungauged) SiteData -- no target, ready to score.

Reuses src.features.recipes.build_feature_frame (the exact assembly the gauged recipes use) with
a weather-derived spine = the TARGET_YEAR daily dates. The SiteData.weather spans TARGET_YEAR
±2 months, so the trailing rolling/lag features have lead-in; trimming the spine to the year
keeps that buffer for lookback while emitting only in-year rows.
"""

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent   # deploy
sys.path.insert(0, str(_HERE.parent))     # repo root -> import src.*
sys.path.insert(0, str(_HERE))            # deploy/ -> sibling build_virtual_basin

from src.features.recipes import build_feature_frame
from build_virtual_basin import TARGET_YEAR


def target_year_spine(site_data, target_year: int = TARGET_YEAR) -> pd.DatetimeIndex:
    """Daily DatetimeIndex over `target_year`, taken from the site's weather dates (which span
    target_year ±2 months). Trimming to the year drops the lookback buffer from the OUTPUT rows
    while keeping it available to the rolling/lag feature computations."""
    wd = pd.DatetimeIndex(sorted(pd.to_datetime(site_data.weather["date"].unique())))
    ystart = pd.Timestamp(f"{target_year}-01-01")
    yend = pd.Timestamp(f"{target_year}-12-31")
    return wd[(wd >= ystart) & (wd <= yend)]


def virtual_recipe(site_data, task: str = "reg", target_year: int = TARGET_YEAR, light: bool = False) -> pd.DataFrame:
    """Feature frame (no target) for a virtual SiteData, task 'reg' or 'clf'.

    Columns match _best_features_REG / _best_features_CLF (recipe_REG/_CLF minus the target), or
    _light_features when `light` is set. Align to a trained model's feature_names before scoring
    (see predict).

    `light` must agree with the model you are about to score: the two column sets differ, and predict() now raises on the mismatch rather than NaN-filling it. It is also what the STATIC widget scores with -- the browser can only assemble the light set -- so the local app passes light=True too, on the same principle as the clientside callbacks: ship the implementation you develop against.
    """
    spine = target_year_spine(site_data, target_year)
    if len(spine) == 0:
        raise ValueError(f"No weather dates fall in {target_year}; the SiteData weather window does not cover it.")
    return build_feature_frame(site_data, task=task, spine=spine, light=light)
