from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

_THIS_DIR = Path(__file__).resolve().parent
_MODELS = _THIS_DIR / "models"

# _ISAAC_MODELS = _ROOT / "isaac" / "models"

from build_virtual_basin import build_virtual_basin
from virtual_recipes import virtual_recipe


def load_model(name: str) -> xgb.Booster:
    """Load a saved XGBoost booster by name"""
    path = _MODELS / f"{name}"
    if not path.exists():
        raise FileNotFoundError(f"No model {path.name} in {_MODELS}.")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def feature_mismatch(booster: xgb.Booster, features: pd.DataFrame) -> dict:
    """Report the column skew between a feature frame and the model."""
    fn = set(booster.feature_names)
    cols = {c for c in features.columns if c != "date"}
    return {"model_only": sorted(fn - cols), "recipe_only": sorted(cols - fn)}


def predict(booster: xgb.Booster, features: pd.DataFrame) -> pd.Series:
    """Score `features` with `booster`, aligning columns to the model's feature_names."""
    fn = booster.feature_names
    date = pd.DatetimeIndex(pd.to_datetime(features["date"])) if "date" in features.columns else None
    X = features.reindex(columns=fn)
    dmat = xgb.DMatrix(X, feature_names=fn)
    pred = booster.predict(dmat)
    return pd.Series(pred, index=date, name="prediction")


# put your regression feature builder code here
# should return a pd.DataFrame whose columns match the columns you trained
# the model with
def build_features_reg(site_data):
    pass


# put your classification feature builder code here
# should return a pd.DataFrame whose columns match the columns you trained
# the model with
def build_features_clf(site_data):
    pass


def test():
    lat = 43.122036
    lon = -91.911358
    year = 2017

    # put the filenames of your reg/clf models from deploy/models/ here
    # can use the existing isaac_*.json models as placeholders to get the
    # test to run
    reg_model_name = "yourmodel.json"  # can set to isaac_REG2.json
    clf_model_name = "yourmodel.json"  # can set to isaac_CLF2.json

    sd = build_virtual_basin(lat=lat, lon=lon, target_year=year)

    reg_f = build_features_reg(site_data=sd, task="reg", target_year=year)
    clf_f = build_features_clf(site_data=sd, task="clf", target_year=year)

    model_reg = load_model(reg_model_name)
    model_clf = load_model(clf_model_name)

    Y_reg = predict(booster=model_reg, features=reg_f)
    Y_clf = predict(booster=model_clf, features=clf_f)

    # the rest of this is just plotting code
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
