"""Parity + correctness tests for the virtual-site deploy path.

Run directly (`python test_parity.py`) or under pytest. The network test (grid reconstruction
via NLDI) is skipped unless DEPLOY_TEST_NETWORK=1 is set.

What these lock in:
  1. Feature-construction parity -- feeding a real site's SiteData through the virtual assembly
     reproduces recipes3.recipe_REG/_CLF minus the target, exactly. This is the guarantee that a
     model trained on recipes3 columns receives byte-identical features at deploy time.
  2. Neighbour-nitrate semantics -- the virtual "rest_of_state_nitrate_lag*" column is the mean
     over ALL 85 sites (a virtual site excludes nobody), shares the training column NAME, and is
     ~1% from the excluded-one training value. So no renaming / rescaling is needed.
  3. Grid reconstruction (network) -- build_virtual_basin at a real basin-1 sensor location
     reproduces that site's basin/grid to within a few percent.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DEPLOY = _HERE.parent
_ROOT = _DEPLOY.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DEPLOY))

import virtual_recipes as vr  # noqa: E402
import predict as pr  # noqa: E402
from data import get_data  # noqa: E402
from data.features import daily_nitrate, nitrate_avg_except_this, _state_daily_wide  # noqa: E402

_UID = "WQS0003"  # a small basin with a full nitrate record


def _recipes3():
    return vr.recipes3


def test_feature_parity_reg():
    """virtual assembly on a real site's SiteData == recipe_REG minus the target (same spine)."""
    r3 = _recipes3()
    sd = get_data(_UID)
    spine = daily_nitrate(_UID).index

    # the built feature set with a SiteData object
    virt = r3.build_feature_frame(sd, task="reg", spine=spine)

    # uid path (cached), includes the target
    gauged = r3.recipe_REG(_UID)

    feat_cols = [c for c in gauged.columns if c != "nitrate_con"]
    assert list(virt.columns) == feat_cols, "REG feature columns differ"
    assert (
        virt[feat_cols].reset_index(drop=True).equals(gauged[feat_cols].reset_index(drop=True))
    ), "REG feature values differ"
    print("OK test_feature_parity_reg")


def test_feature_parity_clf():
    r3 = _recipes3()
    sd = get_data(_UID)
    spine = daily_nitrate(_UID).index
    virt = r3.build_feature_frame(sd, task="clf", spine=spine)
    gauged = r3.recipe_CLF(_UID)
    feat_cols = [c for c in gauged.columns if c != "violation"]
    assert list(virt.columns) == feat_cols, "CLF feature columns differ"
    assert (
        virt[feat_cols].reset_index(drop=True).equals(gauged[feat_cols].reset_index(drop=True))
    ), "CLF feature values differ"
    print("OK test_feature_parity_clf")


def test_neighbour_nitrate_is_all_85_and_same_name():
    """The virtual (sentinel-uid) neighbour feature = mean over ALL 85 sites, shares the training
    column name, and sits ~1% from any single-site-excluded training value."""
    wide = _state_daily_wide()
    n_sites = wide.shape[1]
    all85 = wide.mean(axis=1).asfreq("D").shift(1)  # mean over every site, lag 1
    virt = nitrate_avg_except_this("VIRTUAL", shift=1)  # sentinel excludes nobody
    excl = nitrate_avg_except_this(_UID, shift=1)  # a real training site (excludes itself)

    # same column name for training and virtual (nothing to rename)
    assert virt.name == excl.name == "rest_of_state_nitrate_lag1", "neighbour column name mismatch"

    # virtual == mean over all sites
    a, b = virt.align(all85, join="inner")
    assert np.allclose(a.dropna(), b.loc[a.dropna().index], atol=1e-9), "virtual neighbour != all-85 mean"

    # virtual is within ~1/n of the excluded-one training value (same scale/distribution)
    v, e = virt.align(excl, join="inner")
    m = v.notna() & e.notna()
    rel = (v[m] - e[m]).abs() / (e[m].abs() + 1e-9)
    assert rel.median() < 2.0 / n_sites, f"virtual vs excluded-one gap too large (median {rel.median():.4f})"
    print(f"OK test_neighbour_nitrate_is_all_85_and_same_name (n_sites={n_sites}, median rel diff {rel.median():.4%})")


def test_predict_alignment():
    """predict() aligns any recipe frame to the model feature_names and returns finite scores."""
    r3 = _recipes3()
    sd = get_data(_UID)
    spine = daily_nitrate(_UID).index
    frame = r3.build_feature_frame(sd, task="reg", spine=spine)
    booster = pr.load_model("recipe_REG1.1")
    mm = pr.feature_mismatch(booster, frame)
    # every recipe column is known to the model (no stray columns); model extras are the known skew
    assert mm["recipe_only"] == [], f"recipe produced columns the model never saw: {mm['recipe_only']}"
    pred = pr.predict(booster, frame)
    assert len(pred) == len(frame) and np.isfinite(pred.to_numpy()).all(), "predictions missing/non-finite"
    print(f"OK test_predict_alignment (model_only skew = {mm['model_only']})")


def test_grid_reconstruction_network():
    """build_virtual_basin at a real basin-1 sensor location reproduces its basin/grid (~few %)."""
    if os.environ.get("DEPLOY_TEST_NETWORK") != "1":
        print("SKIP test_grid_reconstruction_network (set DEPLOY_TEST_NETWORK=1 to run)")
        return
    import build_virtual_basin as bvb
    from data.access import get_basin_area

    uid = "WQS0115"  # a basin-1 (NLDI) site
    real = get_data(uid)
    lon, lat = real.sensor_location
    sd = bvb.build_virtual_basin(lat, lon, target_year=2017)
    ratio = sd.basin_area / get_basin_area(uid)
    overlap = len(set(sd.grid.global_node_id) & set(real.grid.global_node_id)) / len(sd.grid)
    assert 0.9 < ratio < 1.1, f"basin area ratio {ratio:.3f} out of tolerance"
    assert overlap > 0.95, f"cell overlap {overlap:.2%} too low"
    print(f"OK test_grid_reconstruction_network (area ratio {ratio:.3f}, cell overlap {overlap:.1%})")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    test_feature_parity_reg()
    test_feature_parity_clf()
    test_neighbour_nitrate_is_all_85_and_same_name()
    test_predict_alignment()
    test_grid_reconstruction_network()
    print("\nall parity tests passed")
