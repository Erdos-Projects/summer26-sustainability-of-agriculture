"""The virtual-site nitrate forecast in Python: a dropped pin -> build_virtual_basin (NLDI basin over grid_global) -> virtual_recipe -> predict, giving the predicted daily nitrate + P(violation) timeseries at that ungauged point for a chosen year.

THE PANEL NO LONGER CALLS THIS. The forecast runs in the browser (assets/clientside/forecast.js), against precomputed reach rows, so that the local app and the published static site take the same path. What survives here is the REFERENCE implementation: it calls NLDI live and rebuilds the features from scratch, which is what makes it worth checking the browser against -- the parity harness scores the same reach both ways and compares.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]  # repo root/
for _p in (_ROOT, _ROOT / "deploy"):
    sys.path.insert(0, str(_p))

from build_virtual_basin import build_virtual_basin  # deploy/
from predict import load_model, load_meta, threshold_for_beta, predict as _predict  # deploy/
from virtual_recipes import virtual_recipe  # deploy/


@dataclass
class VirtualForecast:
    """The predicted timeseries + summary for a virtual (ungauged) site."""
    reg: pd.Series      # predicted nitrate (mg/L), indexed by date
    clf: pd.Series      # P(violation >= 10 mg/L), indexed by date
    peak_prob: float
    days_over: int      # days with predicted nitrate >= 10 mg/L
    basin_geojson: dict
    # β operating point (from the deployed classifier's tuned beta_table; None if the model is untuned)
    beta: float = None          # the recall/precision emphasis the user dialled
    tau: float = None           # decision threshold on P(violation): alarm where clf >= tau
    alarms: pd.Series = None    # bool mask (clf >= tau), aligned to clf.index
    recall: float = None        # OOF catch rate at this operating point (% of violations caught)
    fdr: float = None           # OOF false-discovery rate (% of alarms that are false = 1 - precision)
    base_rate: float = None     # pooled violation prevalence (the FDR is quoted "at ~this prevalence")


def forecast_virtual_site(lat: float, lon: float, target_year: int, beta: float = 2.0, light: bool = True) -> VirtualForecast:
    """Delineate the basin at (lat, lon), build its features for `target_year`, score both models,
    and apply the β operating point to the classifier: alarm days = P(violation) >= tau(β), where
    tau and its honest recall/FDR come from the deployed model's tuned beta_table (see
    src.models.tune_threshold). The NLDI call + build makes this a several-second operation; wrap
    callers in dcc.Loading.

    `light` defaults to True: the light pair is what the STATIC build can score, since the browser can only assemble that feature set, and running the same pair locally is what keeps the two from diverging. Pass light=False to compare against the full recipe models.

    See Also
    --------
    forecast_site_data : score a basin you already have, skipping the snap and the NLDI call.
    """
    return forecast_site_data(build_virtual_basin(lat=lat, lon=lon, target_year=target_year),
                              target_year, beta=beta, light=light)


def forecast_site_data(sd, target_year: int, beta: float = 2.0, light: bool = True) -> VirtualForecast:
    """Score an already-delineated basin. Same models, same recipes; only the delineation is the caller's.

    Split out for widget/static/check_forecast.py, which hands over the SAME basin the reach row was built from. Going through forecast_virtual_site instead would re-snap the pin, and a pin at a reach outlet resolves to the reach BELOW it -- so the harness would compare two different catchments and read the difference as browser-vs-Python error.
    """
    reg = _predict(load_model(task="reg", light=light), virtual_recipe(sd, task="reg", target_year=target_year, light=light))
    clf = _predict(load_model(task="clf", light=light), virtual_recipe(sd, task="clf", target_year=target_year, light=light))
    basin_geojson = json.loads(sd.basin.to_crs("EPSG:4326").to_json())

    op = threshold_for_beta(load_meta(task="clf", light=light), beta)  # None if the clf model has no beta_table
    tau = alarms = recall = fdr = base_rate = None
    if op is not None:
        tau = op["tau"]
        alarms = clf >= tau
        recall, fdr, base_rate = op["recall"], op["fdr"], op["base_rate"]

    return VirtualForecast(
        reg=reg,
        clf=clf,
        peak_prob=float(clf.max()) if len(clf) else float("nan"),
        days_over=int((reg >= 10).sum()),
        basin_geojson=basin_geojson,
        beta=beta,
        tau=tau,
        alarms=alarms,
        recall=recall,
        fdr=fdr,
        base_rate=base_rate,
    )
