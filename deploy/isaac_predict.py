"""Load a trained recipe model and score a (virtual) feature frame.

The saved models (isaac/models/recipe_*.json) are XGBoost boosters. A feature frame from
virtual_recipe / recipes3.build_feature_frame is aligned to the booster's feature_names
before scoring: columns are selected + ordered to match, missing columns are NaN-filled
(XGBoost handles NaN natively -- e.g. a small basin lacking the far `_b1` distance bucket),
and extra recipe columns not in the model are dropped. feature_mismatch() surfaces both sets
so a recipe<->model version skew is explicit rather than silent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]
_MODELS = _THIS_DIR / "models"

# _ISAAC_MODELS = _ROOT / "isaac" / "models"

from build_virtual_basin import build_virtual_basin
from virtual_recipes import virtual_recipe


def load_model(name: str) -> xgb.Booster:
    """Load a saved XGBoost booster by name, e.g. 'recipe_REG2' or 'recipe_CLF2'."""
    path = _MODELS / f"{name}"
    if not path.exists():
        raise FileNotFoundError(f"No model {path.name} in {_MODELS}.")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def feature_mismatch(booster: xgb.Booster, features: pd.DataFrame) -> dict:
    """Report the column skew between a feature frame and the model.

    Returns {'model_only': [...], 'recipe_only': [...]}:
      * model_only  -- features the model expects but the frame lacks (NaN-filled at predict).
      * recipe_only -- features the frame produced but the model never saw (dropped at predict).
    `model_only` that are not just absent distance buckets signals a recipe<->model version skew.
    """
    fn = set(booster.feature_names)
    cols = {c for c in features.columns if c != "date"}
    return {"model_only": sorted(fn - cols), "recipe_only": sorted(cols - fn)}


def predict(booster: xgb.Booster, features: pd.DataFrame) -> pd.Series:
    """Score `features` with `booster`, aligning columns to the model's feature_names.

    Returns a Series of predictions indexed by the frame's `date` column (if present). For a
    classification booster the values are the positive-class probabilities; for regression,
    the predicted target. Column alignment is exactly reindex(model.feature_names): select +
    order to the model, NaN-fill anything missing.
    """
    fn = booster.feature_names
    date = pd.DatetimeIndex(pd.to_datetime(features["date"])) if "date" in features.columns else None
    X = features.reindex(columns=fn)
    dmat = xgb.DMatrix(X, feature_names=fn)
    pred = booster.predict(dmat)
    return pd.Series(pred, index=date, name="prediction")


def test():
    lat = 43.122036
    lon = -91.911358
    ty = 2017

    sd = build_virtual_basin(lat=lat, lon=lon, target_year=ty)
    reg_f = virtual_recipe(site_data=sd, task="reg", target_year=ty)
    clf_f = virtual_recipe(site_data=sd, task="clf", target_year=ty)
    model_reg = load_model("isaac_REG2.json")  # load_model appends ".json"
    model_clf = load_model("isaac_CLF2.json")
    Y_reg = predict(booster=model_reg, features=reg_f)
    Y_clf = predict(booster=model_clf, features=clf_f)

    # basin-wide daily mean precipitation (mean over all basin cells per day), aligned to the
    # prediction dates (the weather itself spans target_year +/- 2 months).
    from data.features import _basin_daily_weather

    precip = _basin_daily_weather(site_data=sd)["precip_in_1d"].reindex(Y_reg.index)

    import matplotlib.pyplot as plt

    fig, (ax_reg, ax_clf) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # make the regression plot
    ax_reg.plot(Y_reg.index, Y_reg.to_numpy(), color="red", label="predicted nitrate (mg/L)")
    ax_reg.set_ylabel("predicted nitrate (mg/L)", color="red")
    ax_reg.tick_params(axis="y", labelcolor="red")
    # plot a line for the 10 mg/L violation threshold
    ax_reg.axhline(10, color="red", ls="--", lw=0.8, alpha=0.5)
    ax_reg_p = ax_reg.twinx()
    ax_reg_p.plot(precip.index, precip.to_numpy(), color="blue", alpha=0.6, label="basin-mean precip (in)")
    ax_reg_p.set_ylabel("basin-mean precip (in)", color="blue")
    ax_reg_p.tick_params(axis="y", labelcolor="blue")
    ax_reg.set_title(f"Regression -- virtual site ({lat:.4f}, {lon:.4f}), {ty}")

    # make the classification plot
    ax_clf.plot(Y_clf.index, Y_clf.to_numpy(), color="red", label="P(violation)")
    ax_clf.set_ylabel("P(violation)", color="red")
    ax_clf.tick_params(axis="y", labelcolor="red")
    ax_clf.set_ylim(0, 1)
    ax_clf_p = ax_clf.twinx()
    ax_clf_p.plot(precip.index, precip.to_numpy(), color="blue", alpha=0.6, label="basin-mean precip (in)")
    ax_clf_p.set_ylabel("basin-mean precip (in)", color="blue")
    ax_clf_p.tick_params(axis="y", labelcolor="blue")
    ax_clf.set_title("Classification -- probability of nitrate violation (>10 mg/L)")
    ax_clf.set_xlabel("date")

    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    test()
