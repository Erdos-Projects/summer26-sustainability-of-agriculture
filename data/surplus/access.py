"""Read-only access layer for per-site nitrogen surplus data."""

import base64
import colorsys
import functools
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "surplus_data"

# HSL colorscale endpoints — match widget/colors.py SURPLUS_LOW / SURPLUS_HIGH
_LOW_HSL = (120, 80, 45)  # green
_HIGH_HSL = (0, 80, 45)  # red


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


def _hsl_to_rgb255(h_deg: float, s_pct: float, l_pct: float) -> tuple[int, int, int]:
    """HSL (degrees, %, %) → (R, G, B) in 0–255.  Uses stdlib colorsys (HLS order)."""
    r, g, b = colorsys.hls_to_rgb(h_deg / 360, l_pct / 100, s_pct / 100)
    return int(r * 255), int(g * 255), int(b * 255)


@functools.lru_cache(maxsize=64)  # cache image
def get_surplus_image(site_uid: str, year: int) -> tuple[str, list]:
    """Return (data_url, bounds) for a surplus heatmap PNG.

    data_url is a base64 PNG string ready for dl.ImageOverlay(url=...).
    bounds is [[lat_min, lon_min], [lat_max, lon_max]].

    The image is transparent where no grid pixel falls inside the basin.
    Surplus values are normalized to the min/max of the requested year.
    Result is cached per (site_uid, year).
    """
    df = get_surplus(site_uid)
    pixels = df[df["year"] == year]
    if pixels.empty:
        raise ValueError(f"No surplus data for {site_uid} year {year}")

    # ── Build regular grid indices from projected x/y (EPSG:5070, metres) ──
    xs = np.sort(pixels["x"].unique())
    ys = np.sort(pixels["y"].unique())[::-1]  # descending → north at top

    x_step = float(np.diff(xs).min()) if len(xs) > 1 else 1.0
    y_step = float(np.diff(np.sort(pixels["y"].unique())).min()) if len(ys) > 1 else 1.0

    col_idx = np.round((pixels["x"].values - xs[0]) / x_step).astype(int)
    row_idx = np.round((ys[0] - pixels["y"].values) / y_step).astype(int)

    n_rows, n_cols = len(ys), len(xs)

    # ── Normalise surplus values ─────────────────────────────────────────────
    vals = pixels["surplus_kgha"].values
    lo, hi = vals.min(), vals.max()
    rng = hi - lo if hi != lo else 1.0
    t = np.clip((vals - lo) / rng, 0.0, 1.0)

    # ── Interpolate colorscale ────────────────────────────────────────────────
    h0, s0, l0 = _LOW_HSL
    h1, s1, l1 = _HIGH_HSL

    rgba = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
    h_vals = h0 + (h1 - h0) * t
    s_vals = s0 + (s1 - s0) * t
    l_vals = l0 + (l1 - l0) * t

    for i in range(len(t)):
        r, g, b = _hsl_to_rgb255(h_vals[i], s_vals[i], l_vals[i])
        rgba[row_idx[i], col_idx[i]] = [r, g, b, 204]  # ~80 % opacity

    # ── Encode to base64 PNG ──────────────────────────────────────────────────
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    bounds = [
        [float(pixels["lat"].min()), float(pixels["lon"].min())],
        [float(pixels["lat"].max()), float(pixels["lon"].max())],
    ]
    return data_url, bounds
