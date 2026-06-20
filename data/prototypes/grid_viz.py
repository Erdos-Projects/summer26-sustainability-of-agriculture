"""Zoomed visual of how the three grids relate and aggregate to the rain grid.

Picks a small window inside WQS0039 (a few rain nodes wide) so the individual
30 m crop pixels and 250 m surplus pixels are visible, and shows three panels:

  1. The three raw grids overlaid (crop pixels, surplus pixels, rain nodes).
  2. Nearest-neighbor assignment: every fine pixel coloured by the rain node it
     is closest to (a Voronoi partition), with the Voronoi boundaries drawn.
  3. The resulting aggregation: stacked pixel counts per node.

Run:
    python data/grid_viz.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parents[1]))  # repo root (prototypes live under data/)

from data.rain import get_rain
from data.surplus import get_surplus
from data import basins
from data.crops import make_crops3 as mc
from data.settings import get_config, get_equal_area_crs

SITE = "WQS0039"
YEAR = 2016
ALBERS = get_equal_area_crs()
OUT = _THIS_DIR / "test" / "grid_viz.png"

HALF = 6500  # half-width of the zoom window in metres (13 km box; rain nodes ~4.8 km apart)
CROP_COLORS = {"Corn": "#E8C500", "Soybeans": "#3A8C3A", "other": "#B0B0B0"}

_to_albers = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)


def rain_nodes(site_uid):
    rain = get_rain(site_uid)[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    x, y = _to_albers.transform(rain["lon"].to_numpy(), rain["lat"].to_numpy())
    return pd.DataFrame({"node_id": np.arange(len(rain)), "x": x, "y": y})


def crop_pixels(site_uid, year):
    poly = basins.get_basin(site_uid).to_crs(mc.CDL_CRS).geometry.union_all()
    f = mc._pixel_frame(mc._clip_path(year), poly, year)
    selected = get_config()["crops"]["agg_crops"]
    f["bucket"] = np.where(f["crop_name"].isin(selected), f["crop_name"], "other")
    return f[["x", "y", "bucket"]]


def main():
    selected = get_config()["crops"]["agg_crops"]
    nodes = rain_nodes(SITE)
    tree = cKDTree(nodes[["x", "y"]].to_numpy())
    node_ids = nodes["node_id"].to_numpy()

    # Centre the window on the most central node so it is fully surrounded.
    cx, cy = nodes["x"].mean(), nodes["y"].mean()
    focal = nodes.iloc[((nodes["x"] - cx) ** 2 + (nodes["y"] - cy) ** 2).idxmin()]
    x0, x1 = focal["x"] - HALF, focal["x"] + HALF
    y0, y1 = focal["y"] - HALF, focal["y"] + HALF

    def in_window(df):
        return df[(df["x"] >= x0) & (df["x"] <= x1) & (df["y"] >= y0) & (df["y"] <= y1)]

    # Fine grids, assigned to nearest node (assignment uses ALL nodes; we only
    # plot what falls in the window).
    crops = crop_pixels(SITE, YEAR)
    crops["node_id"] = node_ids[tree.query(crops[["x", "y"]].to_numpy())[1]]
    cw = in_window(crops)

    surplus = get_surplus(SITE)
    surplus = surplus[surplus["year"] == YEAR].copy()
    surplus["node_id"] = node_ids[tree.query(surplus[["x", "y"]].to_numpy())[1]]
    sw = in_window(surplus)

    nw = in_window(nodes)
    vor = Voronoi(nodes[["x", "y"]].to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), constrained_layout=True)

    # ── Panel 1: the three raw grids overlaid ────────────────────────────────
    ax = axes[0]
    for bucket, color in CROP_COLORS.items():
        pts = cw[cw["bucket"] == bucket]
        ax.scatter(pts["x"], pts["y"], s=2, marker="s", color=color, label=f"crop 30 m: {bucket}")
    for _, p in sw.iterrows():  # 250 m surplus pixels as open squares
        ax.add_patch(Rectangle((p["x"] - 125, p["y"] - 125), 250, 250, fill=False, ec="#1f4e8c", lw=0.25))
    ax.scatter(nw["x"], nw["y"], s=260, marker="*", color="red", ec="k", zorder=5, label="rain node (~800 m)")
    ax.set_title(f"1. Three raw grids overlaid ({SITE}, {YEAR})")
    handles = [
        Line2D([], [], marker="s", ls="", color=CROP_COLORS["Corn"], label="crop 30 m: Corn"),
        Line2D([], [], marker="s", ls="", color=CROP_COLORS["Soybeans"], label="crop 30 m: Soybeans"),
        Line2D([], [], marker="s", ls="", color=CROP_COLORS["other"], label="crop 30 m: other"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="#1f4e8c", label="surplus 250 m"),
        Line2D([], [], marker="*", ls="", color="red", mec="k", label="rain node ~800 m"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    # ── Panel 2: nearest-node assignment (Voronoi partition) ─────────────────
    ax = axes[1]
    win_ids = sorted(cw["node_id"].unique())
    cmap = plt.colormaps["tab20"]
    color_of = {nid: cmap(i % 20) for i, nid in enumerate(win_ids)}
    ax.scatter(cw["x"], cw["y"], s=2, marker="s", color=[color_of[n] for n in cw["node_id"]])
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors="k", line_width=1.2)
    ax.scatter(nw["x"], nw["y"], s=260, marker="*", color="white", ec="k", zorder=5)
    for _, n in nw.iterrows():
        ax.annotate(int(n["node_id"]), (n["x"], n["y"]), fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)
    ax.set_title("2. Each pixel coloured by nearest rain node\n(black = Voronoi cell boundaries)")

    # ── Panel 3: the aggregation that falls out of panel 2 ───────────────────
    ax = axes[2]
    agg = cw.groupby(["node_id", "bucket"]).size().unstack("bucket", fill_value=0)
    for c in selected + ["other"]:
        if c not in agg:
            agg[c] = 0
    agg = agg.loc[win_ids, selected + ["other"]]
    bottom = np.zeros(len(agg))
    for c in selected + ["other"]:
        ax.bar(agg.index.astype(str), agg[c], bottom=bottom, color=CROP_COLORS[c], label=c)
        bottom += agg[c].to_numpy()
    ax.set_title("3. Aggregation: crop-pixel counts per node")
    ax.set_xlabel("node_id")
    ax.set_ylabel("pixels assigned")
    ax.legend(fontsize=8)

    for ax in axes[:2]:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("EPSG:5070 x (m)")
    axes[0].set_ylabel("EPSG:5070 y (m)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"window: {2*HALF/1000:.1f} km box, {len(nw)} rain nodes, "
          f"{len(cw):,} crop px, {len(sw)} surplus px")
    print("\nPer-node aggregation in this window:")
    show = agg.copy()
    show["total"] = show.sum(axis=1)
    print(show.to_string())
    print(f"\nFigure written to {OUT}")


if __name__ == "__main__":
    main()
