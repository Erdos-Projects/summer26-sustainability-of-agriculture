"""
nldi_basins.py
==============
Delineate upstream drainage basins from points using the USGS Network-Linked
Data Index (NLDI), and render them on an interactive MapLibre (plotly) map.

Three public functions
----------------------
delineate_basin(lat, lon)   one point        -> GeoDataFrame  (the spatial data)
delineate_basins(sites)     many points      -> GeoDataFrame  (the spatial data)
render_basin_map(basins)    GeoDataFrame     -> plotly Figure (the map)
delineate_and_map(sites)    many points      -> (GeoDataFrame, Figure) in one call

Helpers
-------
add_area_km2(gdf)           add an equal-area drainage-area column (km^2)
save_basins(gdf, path)      write to GeoParquet / GeoPackage / GeoJSON / Shapefile

Notes
-----
* The NLDI is built on NHDPlus V2 (medium-resolution) catchments. A basin therefore
  terminates at the outlet of the local catchment that contains the query point, and
  includes everything upstream of it. This is robust (it won't blow up to a major
  river the way a mis-snapped raindrop trace can), but it is catchment-resolution, not
  the LiDAR resolution IWQIS uses for Iowa. For a boundary that ends *exactly* at a
  mid-catchment point, use the NLDI pygeoapi `splitcatchment` process instead.
* Geometries come back in EPSG:4326, which is what web maps expect, so nothing needs
  reprojecting in order to display. (Reproject to EPSG:3857 only if you later add a
  contextily basemap, or to an equal-area CRS like EPSG:5070 to compute areas.)
"""

from __future__ import annotations
from pathlib import Path

import math
import os
import requests
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
EQUAL_AREA_CRS = "EPSG:5070"  # NAD83 / Conus Albers -- the standard equal-area CRS for CONUS


# --------------------------------------------------------------------------- #
# ISAAC CODE
# --------------------------------------------------------------------------- #

def get_basin_from_file(uid):
    directory = Path("../../data/USGS-NWIS/geometry/")
    template = "_basins.parquet"
    geoframe = gpd.read_parquet(directory / f"{str(uid) + template}")
    return geoframe

# --------------------------------------------------------------------------- #
# helper: drainage area in km^2                                               #
# --------------------------------------------------------------------------- #
def add_area_km2(
    gdf: gpd.GeoDataFrame,
    equal_area_crs: str = EQUAL_AREA_CRS,
    col: str = "area_km2",
) -> gpd.GeoDataFrame:
    """Add a drainage-area column in km^2, measured in an equal-area projection.

    Geometry is reprojected to ``equal_area_crs`` *only* to measure area; the
    returned GeoDataFrame keeps its original CRS and geometry. Idempotent --
    overwrites ``col`` if it already exists.

    Caveat: basins fetched with ``simplified=True`` have generalized boundaries, so
    their area is approximate. Use ``simplified=False`` if you need survey-grade area.
    """
    out = gdf.copy()
    out[col] = gdf.to_crs(equal_area_crs).area / 1e6
    return out


# --------------------------------------------------------------------------- #
# helper: save to disk                                                        #
# --------------------------------------------------------------------------- #
def save_basins(gdf: gpd.GeoDataFrame, path) -> str:
    """Write basins to disk, dispatching on the file extension.

    .parquet            GeoParquet -- recommended: keeps CRS + dtypes, single file,
                        fast to re-read with gpd.read_parquet (needs pyarrow).
    .gpkg               GeoPackage -- single-file, GIS-friendly, no column limits.
    .geojson / .json    GeoJSON (written in EPSG:4326).
    .shp                Shapefile -- lossy: 10-char column names, multi-file sidecars.

    Returns the path written.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".parquet":
        gdf.to_parquet(path)
    elif ext == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    elif ext in (".geojson", ".json"):
        gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    elif ext == ".shp":
        gdf.to_file(path)
    else:
        raise ValueError(
            f"Unsupported extension {ext!r}; use .parquet, .gpkg, .geojson, or .shp"
        )
    return str(path)


# --------------------------------------------------------------------------- #
# 1. point -> spatial data                                                    #
# --------------------------------------------------------------------------- #
def delineate_basin(
    lat: float,
    lon: float,
    site_id=None,
    simplified: bool = True,
    compute_area: bool = True,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Delineate the upstream drainage basin for a single (lat, lon) point.

    Two NLDI calls:
      1. point-in-polygon lookup -> the NHDPlus COMID of the catchment the point is in
      2. upstream accumulation   -> the basin polygon draining to that catchment outlet

    Parameters
    ----------
    lat, lon : float
        Coordinates in decimal degrees (WGS84 / EPSG:4326).
    site_id : optional
        Label carried onto the output row; defaults to ``comid_<COMID>``.
    simplified : bool
        Ask NLDI for a geometry-simplified polygon (smaller, faster to draw).
    compute_area : bool
        Add an ``area_km2`` column (equal-area, EPSG:5070). One extra reprojection.
    timeout : int
        Per-request timeout in seconds.

    Returns
    -------
    GeoDataFrame  (EPSG:4326), one row, columns:
        site_id, comid, site_lat, site_lon, [area_km2,] geometry

    Raises
    ------
    ValueError if the point does not resolve to an NHDPlus catchment (e.g. it is
    outside CONUS or off-network).
    """
    # WKT/GeoJSON coordinate order is lon-then-lat -- easy to flip by accident.
    pos = requests.get(
        f"{NLDI_BASE}/comid/position",
        params={"coords": f"POINT({lon} {lat})", "f": "json"},
        timeout=timeout,
    )
    pos.raise_for_status()
    feats = pos.json().get("features", [])
    if not feats:
        raise ValueError(
            f"No NHDPlus catchment for (lat={lat}, lon={lon}); "
            "point may be outside CONUS or off-network."
        )
    comid = feats[0]["properties"]["comid"]

    resp = requests.get(
        f"{NLDI_BASE}/comid/{comid}/basin",
        params={"f": "json", "simplified": str(simplified).lower()},
        timeout=timeout,
    )
    resp.raise_for_status()
    basin = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    if basin.empty:
        raise ValueError(f"NLDI returned an empty basin for COMID {comid}.")

    basin = basin[["geometry"]].copy()
    basin["site_id"] = str(site_id) if site_id is not None else f"comid_{comid}"
    basin["comid"] = comid
    basin["site_lat"] = lat
    basin["site_lon"] = lon
    if compute_area:
        basin = add_area_km2(basin)

    cols = ["site_id", "comid", "site_lat", "site_lon"]
    if compute_area:
        cols.append("area_km2")
    cols.append("geometry")
    return basin[cols]


# --------------------------------------------------------------------------- #
# 2. spatial data -> map                                                      #
# --------------------------------------------------------------------------- #
def _view_from_bounds(gdf: gpd.GeoDataFrame):
    """Return (center_dict, zoom) that frames all geometries with a little padding.

    Tile maps have no `fitbounds`, so we estimate an initial view from the data
    extent. The log2(360/span) form maps a ~1 degree span to ~zoom 8.5; the -1
    leaves margin so the basin isn't drawn edge-to-edge.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    center = {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}
    span = max(maxx - minx, maxy - miny, 1e-6)
    zoom = math.log2(360.0 / span) - 1.0
    return center, float(min(12.0, max(2.0, zoom)))


def render_basin_map(
    basins: gpd.GeoDataFrame,
    *,
    map_style: str = "carto-positron",
    opacity: float = 0.45,
    show_points: bool = True,
    center=None,
    zoom=None,
) -> go.Figure:
    """Render one or more basins on an interactive map: one color per site + legend.

    Parameters
    ----------
    basins : GeoDataFrame
        Output of ``delineate_basin`` / ``delineate_and_map``. Needs a ``site_id``
        column; ``site_lat`` / ``site_lon`` are used for the pour-point markers.
    map_style : str
        Token-free MapLibre style: "carto-positron" (default), "open-street-map",
        "carto-darkmatter", or "white-bg".
    opacity : float
        Fill opacity of the basin polygons.
    show_points : bool
        Overlay the query points (the outlets each basin drains to).
    center, zoom :
        Override the auto-computed view if you want.

    Returns
    -------
    plotly.graph_objects.Figure
        Call ``.show()`` to display in a notebook, or ``.write_html(path)`` to save.
    """
    basins = basins.reset_index(drop=True).copy()
    basins["site_id"] = basins["site_id"].astype(str)  # categorical -> discrete colors + legend

    if center is None or zoom is None:
        auto_center, auto_zoom = _view_from_bounds(basins)
        center = center if center is not None else auto_center
        zoom = zoom if zoom is not None else auto_zoom

    # Pass the GeoSeries as the geojson and the index as locations; plotly aligns them.
    hover_data = {"comid": True}
    if "area_km2" in basins.columns:
        hover_data["area_km2"] = ":,.1f"
    fig = px.choropleth_map(
        basins,
        geojson=basins.geometry,
        locations=basins.index,
        color="site_id",
        opacity=opacity,
        center=center,
        zoom=zoom,
        map_style=map_style,
        hover_name="site_id",
        hover_data=hover_data,
    )

    if show_points and {"site_lat", "site_lon"}.issubset(basins.columns):
        pts = basins.drop_duplicates("site_id")
        fig.add_trace(
            go.Scattermap(
                lat=pts["site_lat"],
                lon=pts["site_lon"],
                mode="markers",
                marker=dict(size=9, color="black"),
                text=pts["site_id"],
                name="Sites",
                hoverinfo="text",
            )
        )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Site")
    return fig


# --------------------------------------------------------------------------- #
# 3. many points -> spatial data + map, in one call                           #
# --------------------------------------------------------------------------- #
def _iter_sites(sites, lat_col="lat", lon_col="lon", id_col=None):
    """Normalize the many accepted `sites` shapes into (site_id, lat, lon) triples."""
    if isinstance(sites, pd.DataFrame):
        for i, row in sites.iterrows():
            sid = row[id_col] if id_col is not None else f"site_{i}"
            yield sid, float(row[lat_col]), float(row[lon_col])
        return
    if isinstance(sites, dict):
        for sid, (la, lo) in sites.items():
            yield sid, float(la), float(lo)
        return
    # a single (lat, lon) pair?
    if len(sites) == 2 and all(isinstance(v, (int, float)) for v in sites):
        yield "site_0", float(sites[0]), float(sites[1])
        return
    # otherwise a sequence of (lat, lon) or (site_id, lat, lon)
    for i, item in enumerate(sites):
        if len(item) == 3:
            sid, la, lo = item
        else:
            la, lo = item
            sid = f"site_{i}"
        yield sid, float(la), float(lo)


def delineate_basins(
    sites,
    *,
    lat_col="lat",
    lon_col="lon",
    id_col=None,
    simplified=True,
    compute_area=True,
    save_path=None,
) -> gpd.GeoDataFrame:
    """Delineate basins for many points and return just the spatial data (no map).

    `sites` may be:
      * a single (lat, lon) tuple
      * a list of (lat, lon) or (site_id, lat, lon) tuples
      * a dict {site_id: (lat, lon)}
      * a (Geo)DataFrame, addressed via lat_col / lon_col / id_col

    compute_area : bool
        Add an ``area_km2`` column to every basin (equal-area, EPSG:5070).
    save_path : optional
        If given, write the result via ``save_basins`` (extension picks the format).

    Returns
    -------
    GeoDataFrame  (EPSG:4326), one row per resolved site, columns:
        site_id, comid, site_lat, site_lon, [area_km2,] geometry
    Sites that fail to resolve are skipped with a warning rather than aborting the batch.
    """
    parts, failures = [], []
    for sid, la, lo in _iter_sites(sites, lat_col, lon_col, id_col):
        try:
            parts.append(
                delineate_basin(la, lo, site_id=sid, simplified=simplified,
                                compute_area=compute_area)
            )
        except Exception as e:  # noqa: BLE001  (one bad point shouldn't kill the run)
            failures.append((sid, str(e)))

    if not parts:
        raise RuntimeError(f"No basins could be delineated. Failures: {failures}")

    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    if failures:
        print(f"!  {len(failures)} site(s) skipped: {[f[0] for f in failures]}")

    if save_path is not None:
        save_basins(basins, save_path)
        print(f"   saved {len(basins)} basins -> {save_path}")

    return basins


def delineate_and_map(
    sites,
    *,
    lat_col="lat",
    lon_col="lon",
    id_col=None,
    simplified=True,
    compute_area=True,
    render=True,
    save_path=None,
    **map_kwargs,
):
    """Delineate basins for many sites and render them together in one call.

    Thin convenience wrapper: ``delineate_basins`` (the spatial data) followed by
    ``render_basin_map`` (the figure). For data only, call ``delineate_basins``
    directly. Accepts the same `sites` shapes as ``delineate_basins``.

    Returns
    -------
    (basins_gdf, fig)
        ``fig`` is None when ``render=False``. Extra keyword arguments pass through to
        ``render_basin_map`` (map_style, opacity, show_points, center, zoom).
    """
    basins = delineate_basins(
        sites, lat_col=lat_col, lon_col=lon_col, id_col=id_col,
        simplified=simplified, compute_area=compute_area, save_path=save_path,
    )
    fig = render_basin_map(basins, **map_kwargs) if render else None
    return basins, fig


# --------------------------------------------------------------------------- #
# Example                                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Needs network access to the USGS NLDI.
    iowa_sites = {
        "Des Moines": (41.587, -93.625),
        "Cedar Rapids": (41.978, -91.665),
        "Iowa City": (41.661, -91.530),
    }
    basins, fig = delineate_and_map(iowa_sites, save_path="iowa_basins.parquet")
    print(basins[["site_id", "comid", "area_km2"]].to_string(index=False))
    fig.show()
    # later, in the aggregation layer:
    #   import geopandas as gpd
    #   basins = gpd.read_parquet("iowa_basins.parquet")