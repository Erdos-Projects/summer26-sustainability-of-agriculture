"""Seam between the Dash widget and the virtual-site nitrate forecast.

Replaces the old downstream-sites / forecast_exceedance stub with the deploy path: a dropped pin
-> build_virtual_basin (NLDI basin over grid_global) -> virtual_recipe -> predict, giving the
predicted daily nitrate + P(violation) timeseries at that ungauged point for a chosen year.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SUSTAG2 = Path(__file__).resolve().parents[1]  # sustag2/
for _p in (_SUSTAG2, _SUSTAG2 / "deploy"):
    sys.path.insert(0, str(_p))

from build_virtual_basin import build_virtual_basin  # deploy/
from predict import load_model, predict as _predict  # deploy/
from virtual_recipes import virtual_recipe  # deploy/
from src.features.features import _basin_daily_weather


@dataclass
class VirtualForecast:
    """The predicted timeseries + summary for a virtual (ungauged) site."""
    reg: pd.Series      # predicted nitrate (mg/L), indexed by date
    clf: pd.Series      # P(violation >= 10 mg/L), indexed by date
    precip: pd.Series   # basin-mean daily precip (in), same index
    peak_prob: float
    days_over: int      # days with predicted nitrate >= 10 mg/L
    basin_geojson: dict


def forecast_virtual_site(lat: float, lon: float, target_year: int) -> VirtualForecast:
    """Delineate the basin at (lat, lon), build its features for `target_year`, and score both
    models. The NLDI call + build makes this a several-second operation; wrap callers in dcc.Loading.
    """
    sd = build_virtual_basin(lat=lat, lon=lon, target_year=target_year)
    reg = _predict(load_model(task="reg"), virtual_recipe(sd, task="reg", target_year=target_year))
    clf = _predict(load_model(task="clf"), virtual_recipe(sd, task="clf", target_year=target_year))
    precip = _basin_daily_weather(site_data=sd)["precip_in_1d"].reindex(reg.index)
    basin_geojson = json.loads(sd.basin.to_crs("EPSG:4326").to_json())
    return VirtualForecast(
        reg=reg,
        clf=clf,
        precip=precip,
        peak_prob=float(clf.max()) if len(clf) else float("nan"),
        days_over=int((reg >= 10).sum()),
        basin_geojson=basin_geojson,
    )
