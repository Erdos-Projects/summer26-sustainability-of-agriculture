"""Read-only access layer for per-site nitrogen surplus data."""

import base64
import functools
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "surplus_data"
_RAW_DIR = _THIS_DIR / "surplus_raw"
_META_DIR = _THIS_DIR / "surplus_meta"

_CMAP = colormaps["YlOrRd"]

_MIN_SURPLUS = None
_MAX_SURPLUS = None


def _min_surplus():
    global _MIN_SURPLUS
    if _MIN_SURPLUS is None:
        _MIN_SURPLUS = get_stats()["min_surplus_kgha"].min()
    return _MIN_SURPLUS


def _max_surplus():
    global _MAX_SURPLUS
    if _MAX_SURPLUS is None:
        _MAX_SURPLUS = get_stats()["max_surplus_kgha"].max()
    return _MAX_SURPLUS


def get_surplus(site_uid: str) -> pd.DataFrame:
    """Return the surplus DataFrame for a single site.

    Columns: year, surplus_kgha, total_kg_N, x, y, lon, lat

    Raises FileNotFoundError if the parquet hasn't been generated yet.
    Run make_surplus.py to build it.
    """
    path = _DATA_DIR / f"{site_uid}_surplus.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No surplus data for {site_uid}. Run make_surplus.py to generate it.")
    return pd.read_parquet(path)


def get_all_surplus():
    """Return the full surplus DataFrame."""
    return pd.read_parquet(_RAW_DIR / "iowa_nitrogen_surplus.parquet")


def get_stats():
    """Return the surplus statistics as a DataFrame.

    Returns
    -------
    pandas.DataFrame
        the surplus statistics
    """
    return pd.read_csv(_META_DIR / "surplus_stats.csv")


def get_surplus_image(year: int, df: pd.DataFrame = None, site_uid: str = "", use_global_min_max=True):
    """Return (image, bounds) for a surplus heatmap.

    Parameters
    ----------
    year : int
        The year to render.
    df : pd.DataFrame, optional
        Pre-loaded surplus DataFrame. If None, loaded from disk using site_uid.
    site_uid : str, optional
        Site identifier used to load data when df is not provided.
    use_global_min_max : bool
        If True (default), normalise values to the global min/max from
        surplus_stats.csv so colors are consistent across sites and years.
        If False, normalise to the min/max of the supplied data.

    Returns
    -------
    tuple[PIL.Image.Image, list]
        A PIL RGBA image and bounds [[lat_min, lon_min], [lat_max, lon_max]].
    """
    if df is None:
        if site_uid == "":
            raise ValueError(f"Must provide either df or site_uid")
        else:
            df = get_surplus(site_uid=site_uid)

    pixels = df[df["year"] == year]
    if pixels.empty:
        raise ValueError(f"No surplus data for year {year}")

    # ── Build grid indices in Albers (x, y) space ────────────────────────────
    # Albers x/y (EPSG:5070) are on a regular rectilinear grid; lon/lat are NOT
    # — the same Albers column has slightly different longitudes at each latitude
    # due to the projection, so np.diff(lons).min() returns a near-zero inter-row
    # wobble rather than the actual column spacing, which breaks col_idx mapping.
    xs = np.sort(pixels["x"].unique())
    ys = np.sort(pixels["y"].unique())[::-1]  # descending → north at top
    n_rows, n_cols = len(ys), len(xs)

    col_idx = pd.Categorical(pixels["x"], categories=xs).codes
    row_idx = pd.Categorical(pixels["y"], categories=ys).codes

    # ── Normalise surplus values ─────────────────────────────────────────────
    vals = pixels["surplus_kgha"].values
    if use_global_min_max:
        lo, hi = _min_surplus(), _max_surplus()
    else:
        lo, hi = vals.min(), vals.max()
    rng = hi - lo if hi != lo else 1.0
    t = np.clip((vals - lo) / rng, 0.0, 1.0)

    # ── Apply colormap and set alpha ─────────────────────────────────────────
    rgba_vals = (_CMAP(t) * 255).astype(np.uint8)
    rgba_vals[:, 3] = np.where(vals < lo, 0, 204).astype(np.uint8)

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
    rgba[row_idx, col_idx] = rgba_vals

    img = Image.fromarray(rgba, mode="RGBA")

    bounds = [
        [float(pixels["lat"].min()), float(pixels["lon"].min())],
        [float(pixels["lat"].max()), float(pixels["lon"].max())],
    ]
    return img, bounds


def get_iowa_surplus_image(year: int) -> tuple[Image.Image, list]:
    """Return (image, bounds) for the pre-generated Iowa surplus heatmap.

    Raises FileNotFoundError if the PNG has not been generated yet.
    Run make_surplus.py to generate it.
    """
    iowa_img_path = _RAW_DIR / f"iowa_surplus_{year}.png"
    iowa_bounds_path = _RAW_DIR / f"iowa_surplus_{year}.json"

    if not iowa_img_path.exists() or not iowa_bounds_path.exists():
        raise FileNotFoundError(f"Iowa surplus image for {year} not found. Run make_surplus.py to generate it.")

    with open(iowa_bounds_path) as f:
        bounds = json.load(f)["bounds"]
    return Image.open(iowa_img_path), bounds


@functools.lru_cache(maxsize=18)
def get_iowa_surplus_image_buffer(year: int) -> tuple[str, list]:
    """Return (data_url, bounds) for the full Iowa surplus heatmap PNG.

    Cached per year. Generates and caches the PNG to disk on first call;
    subsequent calls load from disk.
    """
    img, bounds = get_iowa_surplus_image(year)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return data_url, bounds


@functools.lru_cache(maxsize=64)
def get_surplus_image_buffer(site_uid: str, year: int) -> tuple[str, list]:
    """Return (data_url, bounds) for a surplus heatmap PNG.

    data_url is a base64 PNG string ready for dl.ImageOverlay(url=...).
    bounds is [[lat_min, lon_min], [lat_max, lon_max]].

    The image is transparent where no grid pixel falls inside the basin.
    Surplus values are normalized to the global min/max across all sites and
    years (from surplus_stats.csv), so colors are consistent across images.
    Result is cached per (site_uid, year).
    """

    # ── Encode to base64 PNG ──────────────────────────────────────────────────
    img, bounds = get_surplus_image(year=year, site_uid=site_uid)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return data_url, bounds
