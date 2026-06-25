import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import sys

# ── paths (all relative to this script's directory: data/rain/) ───────────────
_THIS_DIR = Path(__file__).resolve().parent
_AUX_DATA_DIR = _THIS_DIR / "aux_data"  # output: derived aux artifacts
_GRAPH_FILE = _AUX_DATA_DIR / "basin_containment_graph.parquet"  # child -> parent containment edges

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data import get_site_ids, get_data, get_global_grid, get_basin_area
from data.basins import get_basin
from data.water import get_location
from data.settings import get_config, get_equal_area_crs


def build_basin_graph3():
    """Directed basin-containment graph keyed on sensor location.

    Adds a directed edge A -> B when site A's monitoring location (lon, lat)
    falls inside site B's (preferred) basin polygon. Self-edges are dropped, so
    A -> B reads as "A's sensor sits inside B's basin" (A is nested in B). A site
    whose sensor lands in no other basin maps to an empty list.

    Both the sensor coordinates (from site_location_metadata.csv) and the basin
    polygons are WGS84 lon/lat (EPSG:4326), so the point-in-polygon test is done
    directly in that CRS via a single vectorized spatial join.

    Returns
    -------
    dict[str, list[str]]
        graph[A] = sorted list of sites B whose basin contains A's sensor.
    """
    sites = get_site_ids()

    points, basins = [], []
    for s in sites:
        try:
            lon, lat = get_location(s)
            points.append({"site_uid": s, "geometry": Point(lon, lat)})
        except (IndexError, KeyError):
            pass  # no sensor coordinates on record

        try:
            basin = get_basin(s).to_crs("EPSG:4326")
            basins.append({"site_uid": s, "geometry": basin.union_all()})
        except (FileNotFoundError, KeyError):
            pass  # no basin polygon on record

    pts_gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
    basin_gdf = gpd.GeoDataFrame(basins, crs="EPSG:4326")

    # return one row for every point, basin pair in which
    # point falls inside the basin
    joined = gpd.sjoin(pts_gdf, basin_gdf, predicate="within", how="left")

    graph = {s: [] for s in sites}
    for a, b in zip(joined["site_uid_left"], joined["site_uid_right"]):
        if pd.isna(b) or a == b:
            continue
        graph[a].append(b)

    return {s: sorted(set(v)) for s, v in graph.items()}


def save_basin_graph():
    """Persist the basin-containment graph as an edge-list parquet.

    Flattens build_basin_graph3() into one row per (child, parent) containment
    edge and writes it to aux_data/basin_graph.parquet. Each edge carries
    parent_area (the enclosing basin's area, m^2) so the full relation can later
    be reduced to an immediate-parent forest by taking, per child, the parent of
    minimum area. Childless (root) sites are omitted from the edge list; the
    loader re-seeds the full node set from get_site_ids().

    Returns the written DataFrame.
    """
    graph = build_basin_graph3()

    # cache each basin's area once; it is reused across many edges
    area = {s: get_basin_area(s) for s in {p for ps in graph.values() for p in ps}}

    rows = [
        {"child": child, "parent": parent, "parent_area": area[parent]}
        for child, parents in graph.items()
        for parent in parents
    ]
    edges = pd.DataFrame(rows, columns=["child", "parent", "parent_area"])

    _AUX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(_GRAPH_FILE, index=False)
    print(f"wrote {len(edges)} containment edges -> {_GRAPH_FILE}")
    return edges


def main(api_keys=None, force: bool = False):
    """Build the auxiliary derived artifacts (the basin-containment graph).

    Pipeline entry point called by data/make_data.py. The graph is cheap to
    rebuild and depends on the basins + grids, so it is always refreshed; `force`
    is accepted for interface compatibility.
    """
    save_basin_graph()


if __name__ == "__main__":
    main()
