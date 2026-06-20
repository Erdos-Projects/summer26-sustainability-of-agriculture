"""Prototype: aggregate crops + surplus onto the rain grid via area-weighted overlap.

Target grid = the rain grid (coarsest, ~4.3 km IEM/Stage-IV cells). For a test
basin (WQS0039) this builds three parquets in data/test/:

    grid_legend.parquet   one row per rain node: node_id, x, y, lat, lon, cell_area
    agg_surplus.parquet   one row per (node_id, year): area-weighted surplus_kgha
                          (intensive mean), total_kg_N (mean x full cell area, so
                          uncovered border area is filled with the in-cell mean),
                          and coverage_frac
    agg_crop.parquet      one row per (node_id, year): pixel counts per crop in
                          [crops].agg_crops, plus "other" (nearest-node, no
                          fractional split — the 30 m correction is sub-percent)

Method
------
Rain cells are the Voronoi cells of the IEM nodes. To bound the basin's edge
cells with *real* neighbours (not an arbitrary clip), we Voronoi a padded halo
of nodes pulled fresh from one IEM daily grid, then drop the infinite (hull)
cells. Target cells are the finite cells whose node intersects the basin.

  - crops:   each 30 m pixel -> nearest node (KD-tree over the full halo, so
             pixels beyond the basin edge correctly fall to halo nodes and are
             dropped); count per (target node, year, crop bucket).
  - surplus: each 250 m cell is a square; gpd.overlay against the target Voronoi
             cells gives area(D n R). Intensive surplus_kgha is the area-weighted
             mean over the covered area; extensive total_kg_N = mean x full cell
             area (== the conservative area-weighted sum where fully covered).

Run:
    python data/grid_test.py
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, box
from scipy.spatial import Voronoi, cKDTree

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parents[1]))  # repo root (prototypes live under data/)
sys.path.insert(0, str(_THIS_DIR.parent / "surplus"))  # so make_surplus's `import gen_surplus_statistics` resolves

from data import basins
from data.crops import make_crops3 as mc
from data.rain import make_rain as mr
from data.surplus import make_surplus as ms
from data.settings import get_config, get_equal_area_crs

SITE = "WQS0067"
ALBERS = get_equal_area_crs()  # "EPSG:5070"
OUT_DIR = _THIS_DIR / "test"

CROP_YEARS = [2008, 2012, 2016]
DISPLAY_YEAR = 2016
IEM_SAMPLE_DATE = date(2018, 6, 15)  # any day with an IEM grid; we only need the cell geometry
SURPLUS_M = 250  # surplus pixel size (EPSG:5070 metres), verified


# ── Rain target grid: padded-halo Voronoi with infinite cells dropped ─────────
def _finite_voronoi(points: np.ndarray) -> dict[int, Polygon]:
    """Map point index -> its Voronoi cell polygon, skipping infinite/empty cells."""
    vor = Voronoi(points)
    polys = {}
    for pidx, ridx in enumerate(vor.point_region):
        verts = vor.regions[ridx]
        if not verts or -1 in verts:
            continue  # hull (infinite) or degenerate cell
        pts = vor.vertices[verts]
        c = pts.mean(axis=0)  # order vertices CCW so the polygon is valid
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys[pidx] = Polygon(pts)
    return polys


def build_rain_grid(site_uid: str):
    """Return (target_gdf, halo_xy, halo_node_id, basin_poly, padded_bbox).

    target_gdf: GeoDataFrame of finite rain cells touching the basin, columns
        node_id, x, y, lat, lon, cell_area, geometry (the Voronoi polygon).
    halo_xy / halo_node_id: every halo node's (x,y) and its node_id (-1 if the
        node is not a kept target), for nearest-node crop assignment.
    """
    basin = basins.get_basin(site_uid).to_crs(ALBERS)
    basin_poly = basin.geometry.union_all()
    minx, miny, maxx, maxy = basin.total_bounds

    # Fresh IEM grid (one day) -> cell polygons + centroids in both CRSs.
    zip_path = mr._download_shapefile(IEM_SAMPLE_DATE)
    day = mr._parse_day_gdf(zip_path, IEM_SAMPLE_DATE)  # EPSG:4326 polygons
    day5070 = day.to_crs(ALBERS)
    centroids = day5070.geometry.centroid  # projected CRS -> accurate centroids
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()
    lonlat = centroids.to_crs("EPSG:4326")  # convert the points (not the polygons) for lon/lat
    lon = lonlat.x.to_numpy()
    lat = lonlat.y.to_numpy()
    grid = gpd.GeoDataFrame(
        {"x": cx, "y": cy, "lon": lon, "lat": lat}, geometry=day5070.geometry.values, crs=ALBERS
    )

    # Nodes whose cell touches the basin are the targets; max spacing sets the pad.
    grid["is_target"] = grid.geometry.intersects(basin_poly)
    tgt_xy = grid.loc[grid["is_target"], ["x", "y"]].to_numpy()
    if len(tgt_xy) < 2:
        # With 0 or 1 target node the k=2 nearest-neighbour query has no neighbour
        # to measure spacing from (it returns inf -> pad inf -> degenerate halo).
        # Fall back to the grid-wide median spacing so a tiny basin still works.
        all_xy = grid[["x", "y"]].to_numpy()
        spacing = float(np.median(cKDTree(all_xy).query(all_xy, k=2)[0][:, 1]))
        pad = 2 * spacing
        print(f"  rain spacing: {len(tgt_xy)} target node(s) — using grid median {spacing:.0f} m -> pad {pad/1000:.1f} km")
    else:
        nn = cKDTree(tgt_xy).query(tgt_xy, k=2)[0][:, 1]
        pad = 2 * float(nn.max())  # 2 x max rain spacing
        print(f"  rain spacing: max {nn.max():.0f} m -> pad {pad/1000:.1f} km; {len(tgt_xy)} target nodes")

    # Halo = every node within the padded bbox (a superset of the targets).
    px0, py0, px1, py1 = minx - pad, miny - pad, maxx + pad, maxy + pad
    halo = grid[grid["x"].between(px0, px1) & grid["y"].between(py0, py1)].reset_index(drop=True)
    polys = _finite_voronoi(halo[["x", "y"]].to_numpy())

    # Keep finite cells whose node intersects the basin; number them.
    halo["node_id"] = -1
    rows = []
    for i in halo.index[halo["is_target"]]:
        if i not in polys:
            print(f"  [warn] target node {i} has an infinite cell (pad too small) — skipped")
            continue
        nid = len(rows)
        halo.at[i, "node_id"] = nid
        rows.append(
            {"node_id": nid, "x": halo.at[i, "x"], "y": halo.at[i, "y"],
             "lat": halo.at[i, "lat"], "lon": halo.at[i, "lon"],
             "cell_area": polys[i].area, "geometry": polys[i]}
        )
    target_gdf = gpd.GeoDataFrame(rows, crs=ALBERS)
    return target_gdf, halo[["x", "y"]].to_numpy(), halo["node_id"].to_numpy(), basin_poly, (px0, py0, px1, py1)


# ── Surplus: fractional area-weighted overlap ────────────────────────────────
def aggregate_surplus(target_gdf, padded_bbox):
    """Area-weighted surplus per (node_id, year). See module docstring for the math."""
    px0, py0, px1, py1 = padded_bbox
    cols = ["pixel_id", "year", "surplus_kgha", "x", "y"]
    merged_file = ms._MERGED_FILE
    merged = pd.read_parquet(merged_file, columns=cols) if merged_file.exists() else ms.build_merged()[cols]
    m = merged[merged["x"].between(px0, px1) & merged["y"].between(py0, py1)].copy()

    # 250 m square per unique surplus pixel (geometry is static across years).
    upx = m.drop_duplicates("pixel_id")[["pixel_id", "x", "y"]].reset_index(drop=True)
    h = SURPLUS_M / 2
    squares = gpd.GeoDataFrame(
        {"pixel_id": upx["pixel_id"]},
        geometry=[box(x - h, y - h, x + h, y + h) for x, y in zip(upx["x"], upx["y"])],
        crs=ALBERS,
    )

    # area(D n R) for each (surplus pixel, rain cell) overlap.
    inter = gpd.overlay(squares, target_gdf[["node_id", "geometry"]], how="intersection")
    inter["area_DR"] = inter.geometry.area

    j = inter.merge(m[["pixel_id", "year", "surplus_kgha"]], on="pixel_id")
    j["w_density"] = j["area_DR"] * j["surplus_kgha"]
    g = (
        j.groupby(["node_id", "year"])
        .agg(covered_area=("area_DR", "sum"), w_density=("w_density", "sum"))
        .reset_index()
        .merge(target_gdf[["node_id", "cell_area"]], on="node_id")
    )
    g["surplus_kgha"] = g["w_density"] / g["covered_area"]  # intensive: area-weighted mean
    g["coverage_frac"] = g["covered_area"] / g["cell_area"]
    g["total_kg_N"] = g["surplus_kgha"] * (g["cell_area"] / 1e4)  # mean x full area (fills border)
    return g[["node_id", "year", "surplus_kgha", "total_kg_N", "coverage_frac"]]


# ── Crops: nearest node (no fractional split) ────────────────────────────────
def aggregate_crops(years, halo_xy, halo_node_id, padded_bbox):
    """Pixel counts per (node_id, year), one column per selected crop + other."""
    selected = get_config()["crops"]["agg_crops"]
    region = box(*padded_bbox)
    tree = cKDTree(halo_xy)
    frames = []
    for year in years:
        clip = mc._clip_path(year)
        if not clip.exists():
            print(f"  (skip crops {year}: no clip)")
            continue
        f = mc._pixel_frame(clip, region, year)  # every pixel in the padded bbox
        f["bucket"] = np.where(f["crop_name"].isin(selected), f["crop_name"], "other")
        f["node_id"] = halo_node_id[tree.query(f[["x", "y"]].to_numpy())[1]]
        frames.append(f.loc[f["node_id"] >= 0, ["node_id", "year", "bucket"]])
    pix = pd.concat(frames, ignore_index=True)

    counts = pix.groupby(["node_id", "year", "bucket"]).size().unstack("bucket", fill_value=0)
    for col in selected + ["other"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[selected + ["other"]].reset_index()
    counts.columns.name = None
    return counts


# ── Display ──────────────────────────────────────────────────────────────────
def summarize(legend, surplus, crops):
    selected = get_config()["crops"]["agg_crops"]
    print(f"\n=== grid_legend ({len(legend)} rain nodes) ===")
    print(legend.drop(columns="geometry").head().to_string(index=False))
    print(f"mean cell area: {legend['cell_area'].mean()/1e6:.2f} km^2")

    sy = surplus[surplus["year"] == DISPLAY_YEAR]
    print(f"\n=== agg_surplus ({len(surplus)} rows = nodes x years) ===")
    print(sy.head().to_string(index=False))
    full = sy[sy["coverage_frac"] >= 0.999]
    print(f"{DISPLAY_YEAR}: total N = {sy['total_kg_N'].sum():,.0f} kg; "
          f"coverage_frac min {sy['coverage_frac'].min():.2f}, "
          f"{len(full)}/{len(sy)} nodes fully covered")

    cy = crops[crops["year"] == DISPLAY_YEAR]
    print(f"\n=== agg_crop ({len(crops)} rows), buckets={selected}+other ===")
    print(cy.head().to_string(index=False))
    totals = cy[selected + ["other"]].sum()
    print(f"{DISPLAY_YEAR} pixel totals: " + ", ".join(f"{k}={int(v):,}" for k, v in totals.items()))


def plot(legend, surplus, crops, basin_poly, path):
    selected = get_config()["crops"]["agg_crops"]
    # Left-merge so every rain cell renders even where surplus is absent (cells
    # outside the Iowa-only surplus domain); those show as grey "no data".
    g = legend.merge(surplus[surplus["year"] == DISPLAY_YEAR], on="node_id", how="left").merge(
        crops[crops["year"] == DISPLAY_YEAR], on="node_id", how="left"
    )
    basin = gpd.GeoSeries([basin_poly], crs=ALBERS)

    panels = [
        ("surplus_kgha", "Surplus (kg/ha), area-weighted mean", "YlOrRd"),
        ("total_kg_N", "Total N (kg), mean x cell area", "YlOrRd"),
        ("coverage_frac", "Surplus coverage fraction", "viridis"),
        (selected[0], f"{selected[0]} pixels per cell", "Greens"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    for ax, (col, title, cmap) in zip(axes.ravel(), panels):
        g.plot(column=col, ax=ax, cmap=cmap, edgecolor="grey", linewidth=0.3, legend=True,
               missing_kwds={"color": "lightgrey", "label": "no surplus (outside Iowa)"})
        basin.boundary.plot(ax=ax, color="black", linewidth=1.2)
        ax.set_title(f"{SITE} {DISPLAY_YEAR}: {title}")
        ax.set_xlabel("EPSG:5070 x (m)")
        ax.set_ylabel("EPSG:5070 y (m)")
        ax.set_aspect("equal")
    fig.savefig(path, dpi=110)
    print(f"\nFigure written to {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building area-weighted common-grid aggregates for {SITE}...")

    target_gdf, halo_xy, halo_node_id, basin_poly, padded_bbox = build_rain_grid(SITE)
    surplus = aggregate_surplus(target_gdf, padded_bbox)
    crops = aggregate_crops(CROP_YEARS, halo_xy, halo_node_id, padded_bbox)

    legend = target_gdf  # carries geometry for plotting; drop it for the parquet
    legend.drop(columns="geometry").to_parquet(OUT_DIR / f"grid_legend_{SITE}.parquet", index=False)
    surplus.to_parquet(OUT_DIR / f"agg_surplus_{SITE}.parquet", index=False)
    crops.to_parquet(OUT_DIR / f"agg_crop_{SITE}.parquet", index=False)

    summarize(legend, surplus, crops)
    plot(legend, surplus, crops, basin_poly, OUT_DIR / f"grid_test_{SITE}.png")


if __name__ == "__main__":
    main()
