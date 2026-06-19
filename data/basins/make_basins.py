"""Compute drainage basins (v1, v2, v3) for all monitoring sites and select
a preferred basin.

Basin types
-----------
basin1 : NLDI position snap
    Calls the USGS NLDI API with the site's (lat, lon), snaps to the enclosing
    NHDPlus catchment (COMID), and returns the full upstream polygon.  Works
    for any site within CONUS.

basin2 : Authoritative source
    USGS sites: queries NLDI by registered NWIS site ID — more accurate than
    the coordinate snap when the gauge is linked to the NHD network.
    IWQIS (WQS) sites: downloads the pre-computed basin KMZ served by the Iowa
    Water Quality Information System.
    Not available for every site (tile-drainage outlets typically lack one).

basin3 : D8 raster BFS flood-fill
    Replicates the IWQIS web-app watershed algorithm from a 500 m resolution
    flow-direction PNG.  Coarser than NLDI polygons; used as a cross-check.

Preferred basin selection
-------------------------
    USGS site AND basin2 exists  →  prefer basin2
    otherwise                    →  prefer basin1

Outputs
-------
Basin parquets  →  data/basins/basin_data/{uid}_basin{1,2,3}.parquet
Metadata table  →  data/basins/preferred_basin.csv
Raster cache    →  data/basins/cache/direction500m.png
Rivers cache    →  data/basins/cache/rivers.gpkg

Usage
-----
    python make_basins.py              # process all sites, skip existing
    python make_basins.py --force      # rewrite all
    python make_basins.py --usgs-only
    python make_basins.py --iwqis-only
"""

import io
import re
import math
import zipfile
import argparse
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import deque
from pathlib import Path
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import Point, shape
from shapely.ops import unary_union

_THIS_DIR       = Path(__file__).resolve().parent
_METADATA       = _THIS_DIR.parent / "water" / "water_meta" / "site_location_metadata.csv"
_BASIN_DATA_DIR = _THIS_DIR / "basin_data"
_CACHE_DIR      = _THIS_DIR / "cache"
_PREFERRED_CSV  = _THIS_DIR / "basin_meta" / "preferred_basin.csv"
_ARCHIVE_CSV    = _THIS_DIR / "basin_meta" / ".preferred_basin_archive.csv"

# Columns that indicate a meaningful change in the archive vs. working CSV.
# Float columns (areas, distances) are excluded — they may drift on recompute.
_KEY_COLS = ["basin_name", "basin_type", "selection_mode", "reviewed"]

EQUAL_AREA_CRS = "EPSG:5070"


# ── Basin 1: NLDI position snap ───────────────────────────────────────────────

_NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"


def _compute_basin1(uid: str, lat: float, lon: float, timeout: int = 60) -> gpd.GeoDataFrame:
    """NLDI position-snap basin for a single site."""
    pos = requests.get(
        f"{_NLDI_BASE}/comid/position",
        params={"coords": f"POINT({lon} {lat})", "f": "json"},
        timeout=timeout,
    )
    pos.raise_for_status()
    feats = pos.json().get("features", [])
    if not feats:
        raise ValueError(f"No NHDPlus catchment near ({lat}, {lon}) for {uid}.")
    comid = feats[0]["properties"]["comid"]

    resp = requests.get(
        f"{_NLDI_BASE}/comid/{comid}/basin",
        params={"f": "json", "simplified": "true"},
        timeout=timeout,
    )
    resp.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    if gdf.empty:
        raise ValueError(f"NLDI returned empty basin for COMID {comid} ({uid}).")

    gdf = gdf[["geometry"]].copy()
    gdf["site_uid"] = uid
    gdf["comid"]    = comid
    gdf["area_km2"] = gdf.to_crs(EQUAL_AREA_CRS).area / 1e6
    return gdf[["site_uid", "comid", "area_km2", "geometry"]]


# ── Basin 2: NLDI nwissite (USGS) or IWQIS KMZ (WQS) ─────────────────────────

_IWQIS_BASE         = "https://iowawis.org/layers/basins"
_IWQIS_STATIONS_URL = "https://iwqis.iowawis.org/app/inc/inc_get_object.php?id=0&subid=0"
_SEARCH_RADIUS_DEG  = 0.03


def _fetch_basin_nwissite(uid: str, timeout: int = 60) -> gpd.GeoDataFrame:
    url  = f"{_NLDI_BASE}/nwissite/{uid}/basin"
    resp = requests.get(url, params={"f": "json", "simplified": "true"}, timeout=timeout)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"NLDI returned no features for {uid}.")
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if gdf.empty:
        raise ValueError(f"NLDI returned empty basin for {uid}.")
    gdf = gdf[["geometry"]].copy()
    gdf["site_uid"] = uid
    gdf["comid"]    = None
    gdf["area_km2"] = gdf.to_crs(EQUAL_AREA_CRS).area / 1e6
    return gdf[["site_uid", "comid", "area_km2", "geometry"]]


def _load_iwqis_station_list() -> pd.DataFrame:
    resp = requests.get(_IWQIS_STATIONS_URL, timeout=120)
    resp.raise_for_status()
    stations = re.findall(r"\[(\d+),([-\d.]+),([-\d.]+),\d+", resp.text)
    df = pd.DataFrame(stations, columns=["id", "lat", "lon"])
    df["id"]  = df["id"].astype(int)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df


def _try_kmz(station_id: int, timeout: int = 30) -> gpd.GeoDataFrame | None:
    resp = requests.get(f"{_IWQIS_BASE}/p{station_id}.kmz", timeout=timeout)
    if not resp.ok:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            kml_name = next((n for n in zf.namelist() if n.endswith(".kml")), None)
            if not kml_name:
                return None
            gdf = gpd.read_file(io.BytesIO(zf.read(kml_name)), driver="KML")
    except Exception:
        return None
    if gdf.empty:
        return None
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf[~gdf["geometry"].is_empty]


def _fetch_basin_iwqis(
    uid: str,
    lat: float,
    lon: float,
    station_df: pd.DataFrame,
    v3_area_km2: float | None = None,
    timeout: int = 30,
) -> gpd.GeoDataFrame:
    numeric_id = int(uid.replace("WQS", ""))
    dist = ((station_df["lat"] - lat) ** 2 + (station_df["lon"] - lon) ** 2) ** 0.5
    nearby = station_df[dist < _SEARCH_RADIUS_DEG].copy()
    nearby["dist"] = dist[nearby.index]
    nearby = nearby.sort_values("dist")

    candidate_ids = list(nearby["id"])
    if numeric_id not in candidate_ids:
        candidate_ids.append(numeric_id)

    best_gdf   = None
    best_score = float("inf")
    site_pt    = Point(lon, lat)

    for sid in candidate_ids:
        raw = _try_kmz(sid, timeout=timeout)
        if raw is None:
            continue
        proj = raw.to_crs(EQUAL_AREA_CRS)
        area = proj.area.sum() / 1e6
        if raw.geometry.union_all().distance(site_pt) > 0.15:
            continue
        centroid = proj.centroid.to_crs("EPSG:4326").iloc[0]
        centroid_dist = ((centroid.y - lat) ** 2 + (centroid.x - lon) ** 2) ** 0.5
        if v3_area_km2 and v3_area_km2 > 1.0:
            ratio = area / v3_area_km2
            if ratio < 0.05 or ratio > 20.0:
                continue
            score = centroid_dist * max(ratio, 1.0 / ratio)
        else:
            score = centroid_dist
        if score < best_score:
            best_score = score
            best_gdf   = raw

    if best_gdf is None:
        raise ValueError(f"No valid IWQIS basin found for {uid}.")

    best_gdf = best_gdf[["geometry"]].copy()
    best_gdf["site_uid"] = uid
    best_gdf["comid"]    = None
    best_gdf["area_km2"] = best_gdf.to_crs(EQUAL_AREA_CRS).area / 1e6
    return best_gdf[["site_uid", "comid", "area_km2", "geometry"]]


def _compute_basin2(
    uid: str,
    lat: float,
    lon: float,
    station_df: pd.DataFrame | None,
    v3_area_km2: float | None,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Authoritative basin: NLDI nwissite (USGS) or IWQIS KMZ (WQS)."""
    if uid.startswith("USGS-"):
        try:
            return _fetch_basin_nwissite(uid, timeout=timeout)
        except requests.HTTPError:
            # Site not linked to NHD by ID; fall back to position snap.
            pos = requests.get(
                f"{_NLDI_BASE}/comid/position",
                params={"coords": f"POINT({lon} {lat})", "f": "json"},
                timeout=timeout,
            )
            pos.raise_for_status()
            feats = pos.json().get("features", [])
            if not feats:
                raise ValueError(f"No NHD catchment near ({lat}, {lon}) for {uid}.")
            comid = feats[0]["properties"]["comid"]
            resp = requests.get(
                f"{_NLDI_BASE}/comid/{comid}/basin",
                params={"f": "json", "simplified": "true"},
                timeout=timeout,
            )
            resp.raise_for_status()
            gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
            if gdf.empty:
                raise ValueError(f"NLDI returned empty basin for COMID {comid} ({uid}).")
            gdf = gdf[["geometry"]].copy()
            gdf["site_uid"] = uid
            gdf["comid"]    = comid
            gdf["area_km2"] = gdf.to_crs(EQUAL_AREA_CRS).area / 1e6
            return gdf[["site_uid", "comid", "area_km2", "geometry"]]
    else:
        return _fetch_basin_iwqis(uid, lat, lon, station_df, v3_area_km2, timeout=timeout)


# ── Basin 3: D8 raster BFS flood-fill ────────────────────────────────────────

_RASTER_URL   = "https://iwqis.iowawis.org/app/inc/watershed/direction500m.png"
_RASTER_CACHE = _CACHE_DIR / "direction500m.png"

_W = 1741
_H = 1057
_RES = 0.004167

_LON_UL    = _RES * (0 - 0.5) - 97.154167 - _RES / 2
_LAT_UL    = 44.53785 + (0 - 0.5) * (-_RES) + _RES / 2
_TRANSFORM = Affine(_RES, 0, _LON_UL, 0, -_RES, _LAT_UL)

_NEIGHBOR_CHECKS = [
    (-1, -1, 3), (0, -1, 2), (+1, -1, 1),
    (-1,  0, 6),              (+1,  0, 4),
    (-1, +1, 9), (0, +1, 8), (+1, +1, 7),
]
_SNAP_RADIUS = 15


def _load_direction_array() -> np.ndarray:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _RASTER_CACHE.exists():
        print("  Downloading direction500m.png...")
        resp = requests.get(_RASTER_URL, timeout=120)
        resp.raise_for_status()
        _RASTER_CACHE.write_bytes(resp.content)
    img       = Image.open(_RASTER_CACHE)
    direction = np.array(img.convert("RGB"))[:, :, 0].astype(np.uint8)
    assert direction.shape == (_H, _W), f"Unexpected raster shape: {direction.shape}"
    return direction


def _ll_to_image_pixel(lat: float, lon: float) -> tuple[int, int]:
    col = int((lon + 97.154167) / _RES + 0.5)
    row = int((44.53785 - lat) / _RES + 0.5)
    return col, row


def _inflow_count(direction: np.ndarray, col: int, row: int) -> int:
    count = 0
    for dc, dr, exp in _NEIGHBOR_CHECKS:
        nc, nr = col + dc, row + dr
        if 0 <= nc < _W and 0 <= nr < _H and direction[nr, nc] == exp:
            count += 1
    return count


def _snap_to_stream(direction: np.ndarray, col: int, row: int) -> tuple[int, int]:
    best_col, best_row = col, row
    best_score = (_inflow_count(direction, col, row), 0)
    r = _SNAP_RADIUS
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            nc, nr = col + dc, row + dr
            if not (0 <= nc < _W and 0 <= nr < _H):
                continue
            if direction[nr, nc] == 0:
                continue
            inflow = _inflow_count(direction, nc, nr)
            dist   = abs(dc) + abs(dr)
            score  = (inflow, -dist)
            if score > best_score:
                best_score = score
                best_col, best_row = nc, nr
    return best_col, best_row


def _bfs(direction: np.ndarray, col: int, row: int) -> np.ndarray:
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[row, col] = 1
    queue = deque([(col, row)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy, expected in _NEIGHBOR_CHECKS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < _W and 0 <= ny < _H and not mask[ny, nx]:
                if direction[ny, nx] == expected:
                    mask[ny, nx] = 1
                    queue.append((nx, ny))
    return mask


def _compute_basin3(uid: str, lat: float, lon: float, direction: np.ndarray) -> gpd.GeoDataFrame | None:
    col, row = _ll_to_image_pixel(lat, lon)
    if not (0 <= col < _W and 0 <= row < _H):
        return None
    if direction[row, col] == 0:
        return None
    mask = _bfs(direction, col, row)
    if mask.sum() < 10:
        snapped_col, snapped_row = _snap_to_stream(direction, col, row)
        if (snapped_col, snapped_row) != (col, row):
            mask = _bfs(direction, snapped_col, snapped_row)
    polys = [shape(geom) for geom, val in shapes(mask, transform=_TRANSFORM) if val == 1]
    if not polys:
        return None
    gdf = gpd.GeoDataFrame(geometry=[unary_union(polys)], crs="EPSG:4326")
    gdf["site_uid"] = uid
    gdf["area_km2"] = gdf.to_crs(EQUAL_AREA_CRS).area / 1e6
    return gdf[["site_uid", "area_km2", "geometry"]]


# ── Distance helper ───────────────────────────────────────────────────────────

def _dist_km(lat: float, lon: float, gdf: gpd.GeoDataFrame) -> float:
    """Shortest distance in km from (lat, lon) to gdf polygon. Returns 0.0 if inside."""
    pt   = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).geometry.iloc[0]
    poly = gdf.to_crs(EQUAL_AREA_CRS).geometry.union_all()
    return pt.distance(poly) / 1000.0


# ── River proximity ───────────────────────────────────────────────────────────

_RIVERS_URL   = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip"
_RIVERS_CACHE = _CACHE_DIR / "rivers.gpkg"


def _load_river_geometry() -> gpd.GeoDataFrame:
    """Return Mississippi + Missouri centerlines projected to EPSG:5070, cached locally."""
    if not _RIVERS_CACHE.exists():
        print("  Downloading Natural Earth 10m rivers...")
        resp = requests.get(_RIVERS_URL, timeout=120)
        resp.raise_for_status()
        tmp = _CACHE_DIR / "_ne_rivers_tmp.zip"
        tmp.write_bytes(resp.content)
        rivers = gpd.read_file(f"/vsizip/{tmp}")
        mis_mo = rivers[rivers["name"].isin(["Mississippi", "Missouri"])].copy()
        mis_mo.to_file(_RIVERS_CACHE, driver="GPKG")
        tmp.unlink()
        print(f"  Cached {len(mis_mo)} river segments to {_RIVERS_CACHE.name}")
    return gpd.read_file(_RIVERS_CACHE).to_crs(EQUAL_AREA_CRS)


# ── Archive helpers ───────────────────────────────────────────────────────────

def _archive_check_parquets(archive: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (site_uid, basin_name) pairs whose parquet is absent from basin_data/."""
    return [
        (r["site_uid"], r["basin_name"])
        for _, r in archive.iterrows()
        if not (_BASIN_DATA_DIR / r["basin_name"]).exists()
    ]


def _archive_print_divergences(archive: pd.DataFrame) -> int:
    """Compare archive to preferred_basin.csv on _KEY_COLS; print any differences.
    Returns the number of diverging sites."""
    current = pd.read_csv(_PREFERRED_CSV)
    merged  = archive.merge(current, on="site_uid", suffixes=("_arch", "_curr"), how="inner")
    diffs: dict[str, list] = {}
    for col in _KEY_COLS:
        a_col, c_col = f"{col}_arch", f"{col}_curr"
        if a_col not in merged.columns or c_col not in merged.columns:
            continue
        mismatch = merged[merged[a_col].astype(str) != merged[c_col].astype(str)]
        for _, r in mismatch.iterrows():
            diffs.setdefault(r["site_uid"], []).append(
                f"{col}: archive={r[a_col]!r}  current={r[c_col]!r}"
            )
    if diffs:
        print(f"\n{len(diffs)} site(s) in preferred_basin.csv diverge from the archive:")
        for uid, changes in diffs.items():
            print(f"  {uid}")
            for c in changes:
                print(f"    {c}")
        print("  → Review these entries; run with --recalculate to rebuild from scratch.")
    return len(diffs)


def _reconstruct_parquet(
    uid: str,
    basin_type: int,
    basin_name: str,
    lat: float,
    lon: float,
    direction: np.ndarray,
    station_df: "pd.DataFrame | None",
) -> bool:
    """Try to reconstruct a missing parquet. Returns True on success."""
    out = _BASIN_DATA_DIR / basin_name
    try:
        if basin_type == 1:
            gdf = _compute_basin1(uid, lat, lon)
        elif basin_type == 2:
            v3_path = _BASIN_DATA_DIR / f"{uid}_basin3.parquet"
            v3_area = float(gpd.read_parquet(v3_path)["area_km2"].iloc[0]) if v3_path.exists() else None
            gdf = _compute_basin2(uid, lat, lon, station_df, v3_area)
        elif basin_type == 3:
            gdf = _compute_basin3(uid, lat, lon, direction)
            if gdf is None:
                print(f"  {uid}: outside Iowa domain, cannot reconstruct basin3.")
                return False
        elif basin_type == 4:
            print(f"  {uid}: basin4 is a custom pin-drop — cannot reconstruct automatically.")
            return False
        else:
            print(f"  {uid}: unknown basin_type {basin_type}, cannot reconstruct.")
            return False
        gdf.to_parquet(out)
        return True
    except Exception as e:
        print(f"  {uid}: reconstruction failed — {e}")
        return False


def _restore_from_archive(
    archive: pd.DataFrame,
    meta: pd.DataFrame,
    direction: np.ndarray,
    station_df: "pd.DataFrame | None",
) -> None:
    """Rebuild preferred_basin.csv from archive, reconstructing any missing parquets.

    For sites where the archived parquet cannot be reconstructed, falls back to
    the auto-selection rules and warns the user.
    """
    coords = meta.set_index("site_uid")[["latitude", "longitude"]].to_dict("index")
    rows = []

    for _, ar in archive.iterrows():
        uid        = ar["site_uid"]
        basin_name = ar["basin_name"]
        basin_type = int(ar["basin_type"])
        path       = _BASIN_DATA_DIR / basin_name

        if not path.exists():
            print(f"  Reconstructing {uid} (basin{basin_type})...")
            if uid not in coords:
                print(f"  {uid}: not in site_location_metadata — skipping.")
                continue
            lat, lon = coords[uid]["latitude"], coords[uid]["longitude"]
            ok = _reconstruct_parquet(uid, basin_type, basin_name, lat, lon, direction, station_df)
            if not ok:
                # Fall back to auto-selection: prefer basin2 for USGS, else basin1
                fallback_type = 2 if uid.startswith("USGS-") and (_BASIN_DATA_DIR / f"{uid}_basin2.parquet").exists() else 1
                fallback_name = f"{uid}_basin{fallback_type}.parquet"
                if (_BASIN_DATA_DIR / fallback_name).exists():
                    print(f"  {uid}: falling back to basin{fallback_type} (auto).")
                    ar = ar.copy()
                    ar["basin_name"]     = fallback_name
                    ar["basin_type"]     = fallback_type
                    ar["selection_mode"] = "auto"
                    ar["reviewed"]       = False
                else:
                    print(f"  {uid}: no fallback parquet available — skipping row.")
                    continue

        rows.append(ar.to_dict())

    df = pd.DataFrame(rows)
    df.to_csv(_PREFERRED_CSV, index=False)
    print(f"Restored preferred_basin.csv with {len(df)} entries from archive.")


# ── Preferred basin metadata ──────────────────────────────────────────────────

def _build_preferred_csv(
    meta: pd.DataFrame,
    rivers_proj: gpd.GeoDataFrame,
    archive: "pd.DataFrame | None" = None,
) -> None:
    """Select preferred basin per site, compute distances/areas/flags, write CSV.

    If an archive is supplied, its basin_type/basin_name/selection_mode/reviewed
    values override the auto-selection rules for sites that appear in the archive.
    New sites (not in the archive) always use auto-selection.
    """
    rivers_union = rivers_proj.geometry.union_all()
    arch_lookup  = {} if archive is None else archive.set_index("site_uid").to_dict("index")
    rows = []

    for _, site_row in meta.iterrows():
        uid = site_row["site_uid"]
        lat, lon = float(site_row["latitude"]), float(site_row["longitude"])

        p1 = _BASIN_DATA_DIR / f"{uid}_basin1.parquet"
        p2 = _BASIN_DATA_DIR / f"{uid}_basin2.parquet"
        p3 = _BASIN_DATA_DIR / f"{uid}_basin3.parquet"

        gdf1 = gpd.read_parquet(p1) if p1.exists() else None
        gdf2 = gpd.read_parquet(p2) if p2.exists() else None
        gdf3 = gpd.read_parquet(p3) if p3.exists() else None

        nan   = float("nan")
        area1 = float(gdf1["area_km2"].sum()) if gdf1 is not None else nan
        area2 = float(gdf2["area_km2"].sum()) if gdf2 is not None else nan
        area3 = float(gdf3["area_km2"].sum()) if gdf3 is not None else nan
        dist1 = _dist_km(lat, lon, gdf1)      if gdf1 is not None else nan
        dist2 = _dist_km(lat, lon, gdf2)      if gdf2 is not None else nan
        dist3 = _dist_km(lat, lon, gdf3)      if gdf3 is not None else nan

        # Preferred basin selection — archive overrides auto-selection
        if uid in arch_lookup:
            ar         = arch_lookup[uid]
            basin_type = int(ar["basin_type"])
            basin_name = ar["basin_name"]
            sel_mode   = ar["selection_mode"]
            reviewed   = ar["reviewed"]
        else:
            basin_type = 2 if (uid.startswith("USGS-") and gdf2 is not None) else 1
            basin_name = f"{uid}_basin{basin_type}.parquet"
            sel_mode   = "auto"
            reviewed   = False

        # Area for flag computation — load custom basin4 if needed
        if basin_type in (1, 2, 3):
            pref_gdf  = {1: gdf1, 2: gdf2, 3: gdf3}[basin_type]
            pref_area = float(pref_gdf["area_km2"].sum()) if pref_gdf is not None else nan
        else:
            p4 = _BASIN_DATA_DIR / basin_name
            pref_area = float(gpd.read_parquet(p4)["area_km2"].sum()) if p4.exists() else nan

        # Flags
        pt_proj = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        ).to_crs(EQUAL_AREA_CRS).geometry.iloc[0]

        flag_area               = (not math.isnan(pref_area)) and pref_area > 50_000
        flag_river              = pt_proj.distance(rivers_union) < 1_000
        existing_dists          = [d for d in [dist1, dist2, dist3] if not math.isnan(d)]
        flag_not_contained      = bool(existing_dists) and all(d > 0 for d in existing_dists)
        flag_basin1_over_basin2 = gdf2 is not None and basin_type == 1

        rows.append(dict(
            site_uid=uid,
            basin_name=basin_name,
            basin_type=basin_type,
            dist_to_1=dist1,
            dist_to_2=dist2,
            dist_to_3=dist3,
            area1=area1,
            area2=area2,
            area3=area3,
            selection_mode=sel_mode,
            reviewed=reviewed,
            flag_area=flag_area,
            flag_river=flag_river,
            flag_not_contained=flag_not_contained,
            flag_basin1_over_basin2=flag_basin1_over_basin2,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(_PREFERRED_CSV, index=False)

    n_flagged = df[["flag_area", "flag_river", "flag_not_contained", "flag_basin1_over_basin2"]].any(axis=1).sum()
    print(f"preferred_basin.csv: {len(df)} sites, {n_flagged} flagged.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    api_keys=None,
    force: bool = False,
    usgs_only: bool = False,
    iwqis_only: bool = False,
    recalculate: bool = False,
) -> None:
    _BASIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    meta   = pd.read_csv(_METADATA)
    coords = meta.set_index("site_uid")[["latitude", "longitude"]].to_dict("index")

    # ── Archive pre-checks ────────────────────────────────────────────────────
    archive = None
    if _ARCHIVE_CSV.exists() and not recalculate:
        archive = pd.read_csv(_ARCHIVE_CSV)

        missing_parquets = _archive_check_parquets(archive)
        divergences = _archive_print_divergences(archive) if _PREFERRED_CSV.exists() else -1

        meta_uids = set(meta["site_uid"])
        arch_uids = set(archive["site_uid"])
        new_sites = meta_uids - arch_uids

        if (not force
                and not missing_parquets
                and divergences == 0
                and not new_sites
                and _PREFERRED_CSV.exists()):
            print("Archive matches preferred_basin.csv, all parquets present — nothing to do.")
            return

        if missing_parquets:
            print(f"\n{len(missing_parquets)} parquet(s) listed in archive are missing from basin_data/:")
            for uid, name in missing_parquets:
                print(f"  {uid}: {name}")
        if new_sites:
            print(f"\n{len(new_sites)} new site(s) not in archive, will compute: {sorted(new_sites)}")

    # ── Load shared resources ─────────────────────────────────────────────────
    all_uids = sorted(meta["site_uid"])
    if iwqis_only:
        uids = [u for u in all_uids if u.startswith("WQS")]
    elif usgs_only:
        uids = [u for u in all_uids if u.startswith("USGS-")]
    else:
        uids = all_uids

    direction = _load_direction_array()

    station_df = None
    if not usgs_only:
        print("Fetching IWQIS station registry (all Iowa sites, used to resolve KMZ IDs)...")
        station_df = _load_iwqis_station_list()
        print(f"  {len(station_df)} stations in registry.\n")

    # ── Restore preferred_basin.csv from archive if missing ───────────────────
    if archive is not None and not _PREFERRED_CSV.exists():
        print("preferred_basin.csv not found — restoring from archive...")
        _restore_from_archive(archive, meta, direction, station_df)

    n = len(uids)

    # ── basin3 first (used by basin2 IWQIS validation) ────────────────────────
    print(f"── Basin 3 (D8 raster) — {n} sites ──")
    ok = skip = fail = 0
    for i, uid in enumerate(uids, 1):
        out    = _BASIN_DATA_DIR / f"{uid}_basin3.parquet"
        prefix = f"  ({i}/{n}) {uid}"
        if out.exists() and not force:
            skip += 1
            continue
        row = coords[uid]
        try:
            gdf = _compute_basin3(uid, row["latitude"], row["longitude"], direction)
            if gdf is None:
                print(f"{prefix}: outside Iowa domain, skipping.")
                skip += 1
                continue
            gdf.to_parquet(out)
            print(f"{prefix}: saved ({gdf['area_km2'].iloc[0]:,.0f} km²)")
            ok += 1
        except Exception as e:
            print(f"{prefix}: failed — {e}")
            fail += 1
    print(f"Basin3: {ok} saved, {skip} skipped, {fail} failed.\n")

    # ── basin1 ────────────────────────────────────────────────────────────────
    print(f"── Basin 1 (NLDI position snap) — {n} sites ──")
    ok = skip = fail = 0
    for i, uid in enumerate(uids, 1):
        out    = _BASIN_DATA_DIR / f"{uid}_basin1.parquet"
        prefix = f"  ({i}/{n}) {uid}"
        if out.exists() and not force:
            skip += 1
            continue
        row = coords[uid]
        try:
            gdf = _compute_basin1(uid, row["latitude"], row["longitude"])
            gdf.to_parquet(out)
            print(f"{prefix}: saved ({gdf['area_km2'].iloc[0]:,.0f} km²)")
            ok += 1
        except Exception as e:
            print(f"{prefix}: failed — {e}")
            fail += 1
    print(f"Basin1: {ok} saved, {skip} skipped, {fail} failed.\n")

    # ── basin2 ────────────────────────────────────────────────────────────────
    print(f"── Basin 2 (authoritative) — {n} sites ──")
    ok = skip = fail = 0
    for i, uid in enumerate(uids, 1):
        out    = _BASIN_DATA_DIR / f"{uid}_basin2.parquet"
        prefix = f"  ({i}/{n}) {uid}"
        if out.exists() and not force:
            skip += 1
            continue
        row = coords[uid]
        lat, lon = row["latitude"], row["longitude"]
        v3_path  = _BASIN_DATA_DIR / f"{uid}_basin3.parquet"
        v3_area  = float(gpd.read_parquet(v3_path)["area_km2"].iloc[0]) if v3_path.exists() else None
        try:
            gdf = _compute_basin2(uid, lat, lon, station_df, v3_area)
            gdf.to_parquet(out)
            print(f"{prefix}: saved ({gdf['area_km2'].iloc[0]:,.0f} km²)")
            ok += 1
        except ValueError as e:
            if force:
                out.unlink(missing_ok=True)
            print(f"{prefix}: no basin found, skipping. ({e})")
            skip += 1
        except Exception as e:
            print(f"{prefix}: failed — {e}")
            fail += 1
    print(f"Basin2: {ok} saved, {skip} skipped, {fail} failed.\n")

    # ── preferred_basin.csv ───────────────────────────────────────────────────
    print("── Building preferred_basin.csv ──")
    rivers_proj = _load_river_geometry()
    _build_preferred_csv(meta, rivers_proj, archive=archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",       action="store_true", help="Rewrite existing parquets.")
    parser.add_argument("--recalculate", action="store_true",
                        help="Recompute all basin geometries from scratch, ignoring the archive.")
    parser.add_argument("--usgs-only",   action="store_true")
    parser.add_argument("--iwqis-only",  action="store_true")
    args = parser.parse_args()
    main(
        force=args.force or args.recalculate,
        recalculate=args.recalculate,
        usgs_only=args.usgs_only,
        iwqis_only=args.iwqis_only,
    )
