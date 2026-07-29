"""Precompute every asset the static widget displays into widget/assets/data/.

The running Dash app reads live through src.data.access, which needs the whole Python stack and ~5 GB of local data. A static build cannot, so everything the site draws has to be captured here first. This script is the single source of that bundle; widget/static/export.py copies what it emits (via manifest.json) into the published site.

WHAT IS *NOT* HERE, deliberately:
  - the statewide surplus heatmap PNGs (18 x 1.7 MB) -- retired from the static build
  - the "bad sites" decorative markers -- presentation-only, no data behind them
  - the NLDI basin polygons and the per-reach feature rows, which are computed by the standalone
    widget/static/fetch_basins.py and widget/static/build_reaches.py. Both take hours and are
    resumable; the `reaches` group here only PACKS what they produced, so changing the shipped
    layout is a re-pack rather than another pass. See widget/static/build_forecast.py.

FACTORIZATION. The two expensive displays are NOT baked as a Cartesian product:
  - rain grid: 83 sites x 18 years x 2 colour modes = 2,988 payloads in the live app. We ship the cell geometry ONCE (global grid), a per-site cell index, and the covariate arrays per year; the browser joins them and computes colour + tooltip.
  - timeseries: reduced to 3 intervals with fixed aggregations (see SERIES_INTERVALS), so it is 83 x 3 series pairs rather than 83 x 7 x 4 x 4.

Usage
-----
    python -m widget.static.build_bundle                 # build everything that is stale
    python -m widget.static.build_bundle --only grid sites
    python -m widget.static.build_bundle --force         # rebuild regardless of the manifest
    python -m widget.static.build_bundle --list          # show artifacts + sizes, build nothing
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent  # widget/static
_WIDGET = _HERE.parent  # widget
_ROOT = _WIDGET.parent  # repo root
sys.path.insert(0, str(_ROOT))

from src.data import access, surplus_viz  # noqa: E402

OUT = _WIDGET / "assets" / "data"
MANIFEST = OUT / "manifest.json"

# ── build knobs ───────────────────────────────────────────────────────────────
# Coordinate rounding, in decimal degrees of precision. 5 dp ~= 1.1 m, far below anything visible at state zoom, and it roughly halves the GeoJSON byte count versus full float repr.
COORD_PRECISION = 5
# Douglas-Peucker tolerance for basin outlines, in degrees (~33 m). Display only -- basin AREAS are read from preferred_basin.csv, never measured from these polygons, so simplification cannot change a number the UI reports.
BASIN_SIMPLIFY_DEG = 0.0003
# NHD display hydrography: same filter/tolerance the widget used at import time.
NHD_MIN_ORDER = 5
NHD_SIMPLIFY_DEG = 0.005

# Years the rain-grid year slider offers. Bounded by surplus_global (2000-2017); crops_global runs to 2025 but is unreachable above 2017 because the slider is shared.
COVARIATE_YEARS = range(2000, 2018)

# Years the forecast dropdown offers. Declared here so the layout reads it from the manifest instead of globbing src/data/interim/weather_global_*.parquet at import time. Bounded above by surplus_global (ends 2017); the lower bound is a shipping choice, since every extra year multiplies the per-reach crop/surplus block across 16,762 reaches.
FORECAST_YEARS = range(2015, 2018)

# Minimum NHD stream order a pin can snap to. Must stay in step with widget/static/fetch_basins.py:
# the snap index and the reach store have to describe the same set, or a pin snaps to a COMID that
# has no feature row. Order >= 3 is 16,762 of the 61,417 reaches, median basin 68 km2, less whatever
# NLDI tombstoned (fetch_basins.tombstoned) -- a reach with no basin leaves the snappable set too.
MIN_STREAM_ORDER = 3

# Days of weather to carry BEFORE the first forecast year. The light recipes shift the weather block back by the basin's travel-time lag (round(median_dist / (2.1 m/s * 86400)), so 0-3 days for Iowa basins), and January rows would otherwise reconstruct against nothing. 75 days is slack, not a tuned number.
WEATHER_BUFFER_DAYS = 75

# (interval, nitrate agg, precip agg). Fixed aggregations -- the two Agg Method dropdowns are retired in the static build, so each site/interval has exactly one series pair.
SERIES_INTERVALS = [("1D", "max", "mean"), ("1W", "max", "mean"), ("1MS", "max", "mean")]

# Crop classes, in the order their counts are packed into crops_{year}.bin.
CROP_CLASSES = ["Alfalfa", "Corn", "Fallow", "Hay_Pasture", "Nonag", "Other", "Small_Grains", "Soybeans"]

BASIN_TYPES = (0, 1, 2, 3)
_EPOCH = pd.Timestamp("1970-01-01")


# ──
# helpers ───────────────────────────────────────────────────────────────────


def _round_coords(obj, nd=COORD_PRECISION):
    """    Recursively round every number in a GeoJSON coordinate tree.

    geopandas' to_json() writes full float repr (-93.12346000000001); rounding in the emitted JSON is what actually saves the bytes, since shapely.set_precision only snaps the values.
    """
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round_coords(o, nd) for o in obj]
    return obj


def _write_geojson(path: Path, gdf, simplify=None, keep=None):
    """Write a GeoDataFrame as EPSG:4326 GeoJSON with rounded coordinates."""
    g = gdf.to_crs("EPSG:4326").copy()
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify)
    if keep is not None:
        g = g[keep + ["geometry"]]
    fc = json.loads(g.to_json())
    for feat in fc.get("features", []):
        feat["geometry"]["coordinates"] = _round_coords(feat["geometry"]["coordinates"])
        feat.pop("id", None)  # positional index; nothing reads it
    return _write_json(path, fc)


def _write_text(path: Path, text: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return len(text.encode())


def _write_json(path: Path, obj) -> int:
    """    allow_nan=False is the point of this wrapper.

    Python's json.dumps happily writes bare NaN / Infinity tokens, which are NOT valid JSON: the browser's fetch().json() and JSON.parse both reject the whole document. So a single non-finite coordinate does not corrupt one feature, it makes the entire file unreadable in the browser while parsing fine in Python -- a failure that cannot be caught by anything short of loading the artifact in a JS engine. Raising at build time instead."""
    return _write_text(path, json.dumps(obj, separators=(",", ":"), allow_nan=False))


def _write_bin(path: Path, blob: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return len(blob)


def _pack(*arrays) -> bytes:
    """    Length-prefixed little-endian pack: [n:int32] then each array's raw bytes in order.

    Every array must be the same length and already the intended dtype. The reader in JS is a DataView int32 read followed by typed-array views at fixed offsets, so the layout has to stay exactly as documented at each call site.
    """
    n = len(arrays[0])
    for a in arrays:
        assert len(a) == n, f"ragged pack: {len(a)} vs {n}"
    out = struct.pack("<i", n)
    for a in arrays:
        out += np.ascontiguousarray(a).tobytes()
    return out


def _days_since_epoch(idx) -> np.ndarray:
    return ((pd.DatetimeIndex(idx) - _EPOCH) // pd.Timedelta(days=1)).to_numpy(dtype=np.int32)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── artifacts ─────────────────────────────────────────────────────────────────
# Each builder returns {relative_path: byte_size}. Names here are the --only keys.


def build_iowa_outline():
    """    The Iowa state polygon, baked from the Census TIGER download.

    The live widget fetches this zip at MODULE IMPORT (map_common.load_iowa_geojson), so importing the app needs the network. Baking it is what makes the app snapshot-able at all.
    """
    import geopandas as gpd

    states = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip")
    iowa = states[states["NAME"] == "Iowa"][["geometry"]]
    return {"iowa_outline.geojson": _write_geojson(OUT / "iowa_outline.geojson", iowa, keep=[])}


def build_sites():
    """    One row per site: location, nitrate stats, basin selection + flags, and basin_area.

    basin_area is precomputed here because access.get_basin_area runs a full build_site_view (D8 flow field + per-cell intersection) per call -- the live Sites Selected table pays that 83 times to render six columns.
    """
    meta = access.get_metadata()
    stats = access.get_all_stats().set_index("site_uid")
    basins = access.get_basin_metadata().set_index("site_uid")

    rows = []
    for r in meta.itertuples(index=False):
        uid = r.site_uid
        row = {
            "site_uid": uid,
            "lat": float(r.latitude),
            "lon": float(r.longitude),
            "source": "USGS" if str(uid).startswith("USGS") else "IWQIS",
        }
        # `state` is here because basin_editor composes "<river> at <town>, <state>"; omitting it
        # silently drops the state from every site label.
        for col in ("river", "town", "state", "draining_area", "nickname"):
            v = getattr(r, col, None)
            row[col] = None if v is None or (isinstance(v, float) and pd.isna(v)) else v
        if uid in stats.index:
            s = stats.loc[uid]
            row |= {
                "nitrate_sparsity": float(s["nitrate_sparsity"]),
                "start_date": str(s["start_date"])[:10],
                "last_date": str(s["last_date"])[:10],
                "lifespan": float(s["lifespan"]),
            }
        if uid in basins.index:
            b = basins.loc[uid]
            row |= {c: (None if pd.isna(b[c]) else (b[c].item() if hasattr(b[c], "item") else b[c])) for c in b.index}
        try:
            row["basin_area_km2"] = access.get_basin_area(uid) / 1e6  # get_basin_area is m^2
        except Exception:
            row["basin_area_km2"] = None
        rows.append(row)

    return {"sites.json": _write_json(OUT / "sites.json", rows)}


def build_basins():
    """    All four delineations per site plus the dissolved union.

    v1/v2/v3 back the basin editor's comparison table; v0 exists only for the 23 USGS sites. The union is the Explore tab's "Show all basins" layer.
    """
    sizes = {}
    for uid in access.get_site_ids():
        for t in BASIN_TYPES:
            try:
                gdf = access.get_basin(uid, type=t)
            except (FileNotFoundError, IndexError, KeyError):
                continue  # v0 is USGS-only; a missing version is normal, not an error
            rel = f"basins/{uid}_v{t}.geojson"
            sizes[rel] = _write_geojson(OUT / rel, gdf, simplify=BASIN_SIMPLIFY_DEG, keep=[])
    # type=0 addresses the PREFERRED basin (access.get_basin's contract), which is what the "Show basin" layer draws -- distinct from the v0 parquet above.
    for uid in access.get_site_ids():
        try:
            gdf = access.get_basin(uid, type=0)
        except (FileNotFoundError, IndexError, KeyError):
            continue
        rel = f"basins/{uid}_preferred.geojson"
        sizes[rel] = _write_geojson(OUT / rel, gdf, simplify=BASIN_SIMPLIFY_DEG, keep=[])

    sizes["basins/union.geojson"] = _write_geojson(
        OUT / "basins" / "union.geojson", access.get_all_basins_union(), simplify=BASIN_SIMPLIFY_DEG, keep=[]
    )
    return sizes


def build_grid():
    """    The global Voronoi grid ONCE, plus a per-site cell index.

    The live app calls access.get_grid(uid), which returns site-local cells carrying node_id, frac_cell_in_basin and dist_to_sensor. Geometry and cell_area are global; only the last three are per-site. So geometry ships once keyed by global_node_id (22,877 cells) and each site gets a small index that maps its node_ids onto it.
    """
    from src.data import site_view

    sizes = {}

    for uid in access.get_site_ids():
        try:
            g = access.get_grid(uid)
        except (FileNotFoundError, IndexError, KeyError):
            continue
        rel = f"site_cells/{uid}.bin"
        # [n][node_id:i4][global_node_id:i4][frac_cell_in_basin:f4][dist_to_sensor:f4]
        sizes[rel] = _write_bin(
            OUT / rel,
            _pack(
                g["node_id"].to_numpy(np.int32),
                g["global_node_id"].to_numpy(np.int32),
                g["frac_cell_in_basin"].to_numpy(np.float32),
                g["dist_to_sensor"].to_numpy(np.float32),
            ),
        )

    gg = _clipped_grid_geometry(site_view._grid_global())
    sizes["grid.geojson"] = _write_geojson(OUT / "grid.geojson", gg, keep=["global_node_id", "cell_area"])
    return sizes


def _clipped_grid_geometry(gg):
    """    The full tessellation with its perimeter ring trimmed to the IEM domain.

    _make_grid._finite_voronoi already drops the cells whose Voronoi region is genuinely infinite, but the ring of nodes just inside the hull keeps cells that are finite and absurd -- they run out to 4.6e8 m easting, roughly 1e6 km2 apiece. Those coordinates are far outside EPSG:5070's valid area, so the inverse projection to EPSG:4326 returns Infinity, and json.dumps writes that as a bare `Infinity` token. The result parses in Python and is rejected outright by the browser's JSON.parse -- ONE bad cell makes the whole 6 MB file unreadable, and every rain grid silently renders empty.

    Clipping to the node bounding box, padded by half the grid spacing so the edge cells match the interior ones in size, makes all 22,877 finite while leaving every interior cell untouched. The full grid ships because the pin-drop basins will reference cells no current site does.

    cell_area is deliberately NOT recomputed. It is the true tessellation area and is load-bearing: transformers.py weights by it, _make_surplus divides by it, and access.get_basin_area sums it. The 305 trimmed cells therefore report an area larger than the polygon drawn for them, which is the honest reading -- that is the area the model used.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from shapely.geometry import box

    xy = gg[["x", "y"]].to_numpy()
    spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    x0, y0, x1, y1 = *xy.min(axis=0), *xy.max(axis=0)
    pad = spacing / 2
    clip = box(x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    out = gg[["global_node_id", "cell_area", "geometry"]].copy()
    out["geometry"] = out.geometry.intersection(clip)
    assert len(out) == len(gg) and out.geometry.is_empty.sum() == 0, "clipping dropped cells"
    return out


def build_covariates():
    """    Per-year surplus and crop arrays, keyed by global_node_id.

    These feed the rain grid's cell colour and tooltip. Shipping them per year (rather than one big table) means the year slider fetches ~230 KB instead of the whole panel.
    """
    sizes = {}
    surplus = access._surplus_global()
    crops = access._crops_global()

    for year in COVARIATE_YEARS:
        s = surplus[surplus["year"] == year]
        if len(s):
            rel = f"covariates/surplus_{year}.bin"
            # [n][global_node_id:i4][surplus_kgha:f4][total_kg_N:f4]
            sizes[rel] = _write_bin(
                OUT / rel,
                _pack(
                    s["global_node_id"].to_numpy(np.int32),
                    s["surplus_kgha"].to_numpy(np.float32),
                    s["total_kg_N"].to_numpy(np.float32),
                ),
            )
        c = crops[crops["year"] == year]
        if len(c):
            rel = f"covariates/crops_{year}.bin"
            # [n][global_node_id:i4][counts:u4 x 8, in CROP_CLASSES order] uint32, not uint16: the Voronoi cells are far from uniform (median ~20k 30 m CDL pixels, max ~171k), and a single class peaks at 97,406 -- over the uint16 ceiling.
            counts = np.stack([c[k].to_numpy() for k in CROP_CLASSES], axis=1)
            assert counts.max() < 2**32, f"crop count overflows uint32 in {year}"
            sizes[rel] = _write_bin(
                OUT / rel,
                struct.pack("<i", len(c))
                + c["global_node_id"].to_numpy(np.int32).tobytes()
                + counts.astype(np.uint32).tobytes(),
            )
    return sizes


def build_series():
    """    Nitrate + basin-mean precip per (site, interval), at the fixed aggregations.

    The precip half also removes the worst read in the live app: info_panel calls access.get_weather(uid), which for a large basin reads ~12 M weather rows to produce a few thousand daily means, on every graph interaction.

    ONE DELIBERATE DIVERGENCE from the live figure: precip is reindexed onto the nitrate index, so the two series share an x grid. access.get_weather pads the nitrate span by _WEATHER_PAD (60 days each side), so the server-rendered chart draws ~120 precip points beyond where nitrate exists; here it stops with the nitrate. Values on every shared date are identical -- verified against _build_timeseries_figure -- only the padded tails differ.
    """
    sizes = {}
    for uid in access.get_site_ids():
        try:
            w = access.get_weather(uid)
            daily_precip = w.groupby("date")["precip_in_1d"].mean()
            daily_precip.index = pd.DatetimeIndex(daily_precip.index)
        except (FileNotFoundError, KeyError):
            daily_precip = None

        for interval, nit_agg, rain_agg in SERIES_INTERVALS:
            try:
                nitrate = access.aggregate_by_interval(site_uid=uid, interval=interval, agg_func=nit_agg)
            except (FileNotFoundError, KeyError):
                continue
            nitrate.index = pd.DatetimeIndex(nitrate.index).tz_localize(None)
            precip = (
                daily_precip.resample(interval).agg(rain_agg).reindex(nitrate.index)
                if daily_precip is not None
                else pd.Series(np.nan, index=nitrate.index)
            )
            rel = f"series/{uid}_{interval}.bin"
            # [n][days_since_epoch:i4][nitrate:f4][precip:f4]
            sizes[rel] = _write_bin(
                OUT / rel,
                _pack(
                    _days_since_epoch(nitrate.index),
                    nitrate.to_numpy(np.float32),
                    precip.to_numpy(np.float32),
                ),
            )
    return sizes


def build_hydro():
    """    NHD display hydrography, generated here instead of at widget import time.

    map_common._ensure_nhd_assets() does this on import today, reading 58 MB of parquet if the files are absent. Same filter and tolerance, so the output is byte-comparable.
    """
    assets = _WIDGET / "assets"
    sizes = {}

    fl = access.get_flowlines()
    fl = fl[fl["StreamOrde"] >= NHD_MIN_ORDER][["geometry"]]
    sizes["../iowa_flowlines.geojson"] = _write_geojson(
        assets / "iowa_flowlines.geojson", fl, simplify=NHD_SIMPLIFY_DEG, keep=[]
    )
    wb = access.get_waterbodies()[["geometry"]]
    sizes["../iowa_waterbodies.geojson"] = _write_geojson(
        assets / "iowa_waterbodies.geojson", wb, simplify=NHD_SIMPLIFY_DEG, keep=[]
    )
    return sizes


# ── model repack ──────────────────────────────────────────────────────────────
_OBJECTIVE_CODE = {"reg:squarederror": 0, "binary:logistic": 1}


def _pack_booster(model_json: dict) -> bytes:
    """    Repack an XGBoost JSON booster into flat typed arrays the browser can walk.

    6 MB of JSON per model becomes ~675 KB, because everything inference does not read is dropped: loss_changes, sum_hessian, parents, split_type, the four empty categories* arrays, and the whole right_children array. That last one is safe to derive rather than store -- XGBoost allocates children in pairs, and `right == left + 1` holds for all 5,000 trees of both models (asserted below rather than assumed).

    Two details that are easy to get wrong and silently wrong if you do:
      - For a LEAF, split_conditions holds the leaf weight, so one float array serves as both threshold and output. The learning rate is already baked into it; do not rescale.
      - base_score is stored in PREDICTION space, not margin space. reg:squarederror adds it to the margin directly; binary:logistic needs logit() first, then the sigmoid at the end.

    The stride is the widest tree in THIS booster, not a constant: it follows max_depth (15 node slots at depth 3, 63 at depth 5), so a retune to a deeper model changes it. It ships in the header and the browser reads it from there, so nothing but this function needs to know.

    Layout: an int32 header [n_trees, stride, n_features, objective] then float32 base_margin, then four flat arrays of n_trees*stride: split_indices int16, left_children int16 (-1 = leaf), default_left uint8, split_conditions float32.
    """
    learner = model_json["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    n = len(trees)
    stride = max(len(tree["left_children"]) for tree in trees)

    split_idx = np.zeros(n * stride, np.int16)
    left = np.full(n * stride, -1, np.int16)
    default_left = np.zeros(n * stride, np.uint8)
    cond = np.zeros(n * stride, np.float32)

    for t, tree in enumerate(trees):
        L = np.asarray(tree["left_children"], np.int32)
        R = np.asarray(tree["right_children"], np.int32)
        internal = L >= 0
        assert (R[internal] == L[internal] + 1).all(), f"tree {t}: right child is not left+1"
        assert not any(tree["categories"]), f"tree {t} has categorical splits, which the walker does not implement"
        o = t * stride
        k = len(L)
        split_idx[o:o + k] = np.asarray(tree["split_indices"], np.int32).astype(np.int16)
        left[o:o + k] = L.astype(np.int16)
        default_left[o:o + k] = np.asarray(tree["default_left"], np.uint8)
        cond[o:o + k] = np.asarray(tree["split_conditions"], np.float32)

    objective = learner["objective"]["name"]
    if objective not in _OBJECTIVE_CODE:
        raise ValueError(f"unsupported objective {objective!r}; the walker handles {sorted(_OBJECTIVE_CODE)}")
    # learner_model_param values are STRINGS, and base_score arrives as a stringified 1-vector ("[2.5795278E-1]").
    base_score = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
    base_margin = np.log(base_score / (1.0 - base_score)) if objective == "binary:logistic" else base_score

    header = struct.pack(
        "<4if", n, stride, int(learner["learner_model_param"]["num_feature"]),
        _OBJECTIVE_CODE[objective], base_margin,
    )
    return header + split_idx.tobytes() + left.tobytes() + default_left.tobytes() + cond.tobytes()


def build_models():
    """    The light boosters, repacked, plus their feature order and operating points.

    Reads from deploy/models/ under the STABLE light names (see deploy/predict.LIGHT_*_NAME), so promoting a retrain is a copy into that directory and nothing here changes. The sidecar travels alongside as JSON because the browser needs the feature ORDER (the walker indexes columns positionally) and the classifier's beta_table.
    """
    sys.path.insert(0, str(_ROOT / "deploy"))  # deploy/ is a plain directory, not a package
    import predict as deploy_predict

    models_dir = _ROOT / "deploy" / "models"
    sizes = {}
    for task, name in (("reg", deploy_predict.LIGHT_REG_NAME), ("clf", deploy_predict.LIGHT_CLF_NAME)):
        src = models_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} is missing. Train with `python -m src.models.train --light`, then copy the "
                f"booster and its .meta.json into deploy/models/ as {name}."
            )
        meta_path = models_dir / f"{name}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        sizes[f"models/{task}.bin"] = _write_bin(OUT / "models" / f"{task}.bin", _pack_booster(json.loads(src.read_text())))
        sizes[f"models/{task}.json"] = _write_json(
            OUT / "models" / f"{task}.json",
            {k: meta[k] for k in ("feat", "task", "target", "beta_table", "base_rate") if k in meta},
        )
    return sizes


def build_palette():
    """    256-entry LUT for the surplus colour ramp, plus the crop class colours.

    surplus_viz.surplus_to_hex evaluates matplotlib's YlOrRd. Shipping a sampled LUT rather than reimplementing the colormap in JS guarantees the browser draws the exact same colours as the Python path, with no second implementation to keep in sync.
    """
    sys.path.insert(0, str(_WIDGET))
    import colors as widget_colors  # widget/colors.py -- flat import, matches how the app loads it

    lo, hi = float(surplus_viz._min_surplus()), float(surplus_viz._max_surplus())
    lut = [surplus_viz.surplus_to_hex(lo + (hi - lo) * i / 255.0) for i in range(256)]
    payload = {
        "surplus": {"lo": lo, "hi": hi, "lut": lut},
        "crops": {k: widget_colors.CROP_COLORS[k] for k in CROP_CLASSES},
        "nodata": "#cccccc",
    }
    return {"palette.json": _write_json(OUT / "palette.json", payload)}


BUILDERS = {
    "iowa_outline": build_iowa_outline,
    "sites": build_sites,
    "basins": build_basins,
    "grid": build_grid,
    "covariates": build_covariates,
    "series": build_series,
    "hydro": build_hydro,
    "palette": build_palette,
    "models": build_models,
    # Forecast path. Separate groups because the reach pass is hours long and must not be redone
    # when an upstream group changes; see widget/static/build_forecast.py.
    "weather_basis": lambda: _forecast().build_weather_basis(
        OUT, FORECAST_YEARS, WEATHER_BUFFER_DAYS, _days_since_epoch, _write_bin, _ROOT / "src" / "data" / "cache"
    ),
    "cross_site": lambda: _forecast().build_cross_site(
        OUT, FORECAST_YEARS, WEATHER_BUFFER_DAYS, _days_since_epoch, _write_bin
    ),
    "snap_index": lambda: _forecast().build_snap_index(OUT, MIN_STREAM_ORDER, _write_bin),
    # Packs what widget/static/build_reaches.py computed; it does NOT compute. See that script.
    "reaches": lambda: _forecast().build_reaches(
        OUT, FORECAST_YEARS, CROP_CLASSES, _write_bin,
        _ROOT / "src" / "data" / "cache", _ROOT / "src" / "data" / "cache" / "reach_rows",
        _ROOT / "src" / "data" / "raw" / "basins" / "nldi",
    ),
}


def _forecast():
    """Imported lazily: build_forecast pulls in the feature stack, which the light groups do not need."""
    from widget.static import build_forecast

    return build_forecast


# ──
# driver ────────────────────────────────────────────────────────────────────


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"artifacts": {}, "groups": {}}


def main(only=None, force=False, list_only=False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    groups = manifest.setdefault("groups", {})
    artifacts = manifest.setdefault("artifacts", {})

    if list_only:
        total = sum(a["bytes"] for a in artifacts.values())
        for g, files in sorted(groups.items()):
            gb = sum(artifacts[f]["bytes"] for f in files if f in artifacts)
            print(f"{g:14s} {len(files):5d} files  {gb / 1e6:8.2f} MB")
        print(f"{'TOTAL':14s} {len(artifacts):5d} files  {total / 1e6:8.2f} MB")
        return

    todo = list(BUILDERS) if only is None else list(only)
    for name in todo:
        if name not in BUILDERS:
            raise SystemExit(f"unknown artifact group {name!r}; choose from {sorted(BUILDERS)}")
        if not force and name in groups and all((OUT / f).exists() for f in groups[name]):
            print(f"── {name}: up to date ({len(groups[name])} files), skipping")
            continue
        print(f"── {name}: building ...", flush=True)
        sizes = BUILDERS[name]()
        groups[name] = sorted(sizes)
        for rel, nbytes in sizes.items():
            artifacts[rel] = {"bytes": nbytes, "sha": _sha(OUT / rel)}
        print(f"   {len(sizes)} file(s), {sum(sizes.values()) / 1e6:.2f} MB")
        # Checkpoint after EVERY group, not once at the end. export.py copies from the manifest, so a group that built but never got recorded is a file that silently 404s on the static site
        # -- which is exactly what happened when a later group raised mid-run.
        _write_json(MANIFEST, manifest)

    manifest["coverage"] = {
        "covariate_years": [COVARIATE_YEARS.start, COVARIATE_YEARS.stop - 1],
        "forecast_years": [FORECAST_YEARS.start, FORECAST_YEARS.stop - 1],
        "series_intervals": [{"interval": i, "nitrate_agg": n, "precip_agg": p} for i, n, p in SERIES_INTERVALS],
        "crop_classes": CROP_CLASSES,
        "basin_types": list(BASIN_TYPES),
        "n_sites": len(access.get_site_ids()),
    }
    _write_json(MANIFEST, manifest)
    total = sum(a["bytes"] for a in artifacts.values())
    print(f"\nmanifest: {len(artifacts)} artifacts, {total / 1e6:.2f} MB -> {MANIFEST}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="GROUP", help=f"build only these: {sorted(BUILDERS)}")
    p.add_argument("--force", action="store_true", help="rebuild even if the manifest says current")
    p.add_argument("--list", action="store_true", dest="list_only", help="print the current bundle and exit")
    a = p.parse_args()
    main(only=a.only, force=a.force, list_only=a.list_only)
