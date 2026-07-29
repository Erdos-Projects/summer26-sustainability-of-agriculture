"""D8 flow-direction raster primitives (shared by build + runtime).

The IWQIS 500 m flow-direction raster (direction500m.png) and its geo-referencing. Both the
basin builder (basin3 BFS delineation) and the runtime read path (site_view's dist_to_sensor
flow field) need these, so by the sorting rule they live on the DATA side -- src/build may
import this, but nothing here imports src/build.

Each cell stores its downslope neighbour as a numeric-keypad direction (7 8 9 / 4 5 6 / 1 2 3;
0 = nodata). NEIGHBOR_CHECKS maps a neighbour offset to the direction code an UPSTREAM neighbour
must hold to drain into the centre cell.
"""

import sys
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Geod
from rasterio.transform import Affine

_DATA = Path(__file__).resolve().parent               # src/data
_RASTER = _DATA / "raw" / "basins" / "cache" / "direction500m.png"
_RASTER_URL = "https://iwqis.iowawis.org/app/inc/watershed/direction500m.png"
_DERIVED_CACHE = _DATA / "cache"  # gitignored, regenerable derived artifacts (flow accumulation)

# ── geo-referencing of the 500 m raster (verbatim from make_basins) ────────────
W = 1741
H = 1057
RES = 0.004167
LON_UL = RES * (0 - 0.5) - 97.154167 - RES / 2
LAT_UL = 44.53785 + (0 - 0.5) * (-RES) + RES / 2
TRANSFORM = Affine(RES, 0, LON_UL, 0, -RES, LAT_UL)

# neighbour offset (dc, dr) -> direction code an upstream neighbour must hold to flow into centre
NEIGHBOR_CHECKS = [
    (-1, -1, 3), (0, -1, 2), (+1, -1, 1),
    (-1,  0, 6),              (+1,  0, 4),
    (-1, +1, 9), (0, +1, 8), (+1, +1, 7),
]

_DIRECTION: np.ndarray | None = None


def load_direction_array() -> np.ndarray:
    """The D8 direction raster as an (H, W) uint8 array (cached in-process).

    Reads src/data/raw/basins/cache/direction500m.png, else the legacy cache, else downloads it.
    """
    global _DIRECTION
    if _DIRECTION is not None:
        return _DIRECTION

    if not _RASTER.exists():
        import requests
        print("  Downloading direction500m.png...")
        _RASTER.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_RASTER_URL, timeout=120)
        resp.raise_for_status()
        _RASTER.write_bytes(resp.content)

    direction = np.array(Image.open(_RASTER).convert("RGB"))[:, :, 0].astype(np.uint8)
    assert direction.shape == (H, W), f"Unexpected raster shape: {direction.shape}"
    _DIRECTION = direction
    return direction


def ll_to_image_pixel(lat: float, lon: float) -> tuple[int, int]:
    """(lat, lon) -> (col, row) pixel index in the D8 raster."""
    col = int((lon + 97.154167) / RES + 0.5)
    row = int((44.53785 - lat) / RES + 0.5)
    return col, row


# ── flow distance: cell -> sensor outlet, along the D8 drainage network ────────
# Ported from make_grid. Walking UP the D8 tree from a sensor's snapped pour point gives the
# flow distance to every upstream cell in one pass. Used by site_view's dist_to_sensor.

_GEOD = Geod(ellps="WGS84")
# direction code -> downstream (dcol, drow); inverse of NEIGHBOR_CHECKS.
_D8_STEP = {7: (-1, -1), 8: (0, -1), 9: (1, -1), 4: (-1, 0), 6: (1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1)}
_OUTLET_SNAP_RADIUS = 10   # cells (~5 km) searched for the main-stem outlet
_OUTLET_ACC_FRAC = 0.5     # min fraction of window-max accumulation to count as main stem
NODE_SNAP_RADIUS = 4       # cells (~2 km) to recover a node straddling the basin divide

_ACCUM: np.ndarray | None = None

# Flow fields, most-recent-first, keyed on the rounded outlet. The bound is load-bearing: a field is the whole raster in float64 (14.7 MB), and a caller delineating ARBITRARY pins (deploy's virtual basins, widget/static/build_reaches.py) visits a new outlet every time -- unbounded, six workers over ~1,000 basins each reach ~88 GB. A small cap still serves what the cache is for, since the training path groups edges by parent and those calls are consecutive.
_FLOW_FIELD_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_FLOW_FIELD_CACHE_MAX = 4


def _accum_cache_file() -> Path:
    """On-disk path for the cached flow accumulation, keyed on the identity of the raster it derives from. A new or edited raster yields a new filename, so a stale array can never be read back."""
    st = _RASTER.stat()
    return _DERIVED_CACHE / f"d8_accum_{H}x{W}_{st.st_size}_{st.st_mtime_ns}.npy"


def _flow_accumulation(direction: np.ndarray) -> np.ndarray:
    """Upstream-cell count for every D8 cell (computed once, cached grid-wide and on disk).

    A Kahn topological sort over all H*W cells in a pure-Python loop. It is a pure function of the direction raster, so it is memoized to src/data/cache/ and re-read on later processes. That matters most for the deploy/widget path and the static-site reach pass, which delineate arbitrary pins and so can never precompute anything per site, yet still pay this before their first basin.
    """
    global _ACCUM
    if _ACCUM is not None:
        return _ACCUM

    cache = _accum_cache_file()
    if cache.exists():
        try:
            _ACCUM = np.load(cache)
            return _ACCUM
        except Exception as e:
            print(f"  [warn] unreadable flow-accumulation cache {cache.name} ({e}); recomputing.")

    down = np.full((H, W, 2), -1, np.int32)
    indeg = np.zeros((H, W), np.int32)
    for code, (dc, dr) in _D8_STEP.items():
        rs, cs = np.where(direction == code)
        nc, nr = cs + dc, rs + dr
        ok = (nc >= 0) & (nc < W) & (nr >= 0) & (nr < H)
        down[rs[ok], cs[ok], 0] = nc[ok]
        down[rs[ok], cs[ok], 1] = nr[ok]
        np.add.at(indeg, (nr[ok], nc[ok]), 1)
    acc = np.ones((H, W), np.int64)
    q = deque(zip(*np.where(indeg == 0)))
    while q:
        r, c = q.popleft()
        dc, dr = down[r, c]
        if dc < 0:
            continue
        acc[dr, dc] += acc[r, c]
        indeg[dr, dc] -= 1
        if indeg[dr, dc] == 0:
            q.append((dr, dc))
    _ACCUM = acc
    try:
        _DERIVED_CACHE.mkdir(parents=True, exist_ok=True)
        for old in _DERIVED_CACHE.glob("d8_accum_*.npy"):
            old.unlink()  # a superseded raster's array; the filename key makes these dead weight
        np.save(cache, acc)
    except Exception as e:
        print(f"  [warn] could not cache flow accumulation ({e}); it will be recomputed next process.")
    return acc


def _snap_outlet(direction: np.ndarray, col: int, row: int) -> tuple[int, int]:
    """Snap a sensor pixel to the nearest main-stem cell (pour-point snapping)."""
    acc = _flow_accumulation(direction)
    R = _OUTLET_SNAP_RADIUS
    c0, c1 = max(0, col - R), min(W, col + R + 1)
    r0, r1 = max(0, row - R), min(H, row + R + 1)
    sub = acc[r0:r1, c0:c1]
    rs, cs = np.where(sub >= _OUTLET_ACC_FRAC * sub.max())
    cc, rr = cs + c0, rs + r0
    j = int(np.argmin((cc - col) ** 2 + (rr - row) ** 2))
    return int(cc[j]), int(rr[j])


@lru_cache(maxsize=1)
def _hop_lengths() -> np.ndarray:
    """Hop length in metres for each NEIGHBOR_CHECKS direction out of each raster row: H rows x 8 directions.

    The raster is north-up and axis-aligned (TRANSFORM.b == TRANSFORM.d == 0), so a cell centre's latitude is a function of its row alone and a neighbour's longitude offset is a function of dx alone. Geodesic distance on an ellipsoid of revolution is invariant under rotation in longitude, so a hop's length depends only on (row, dx, dy) -- never on the column. That collapses the BFS's per-cell-per-neighbour geodesic solve into one table of H x 8 entries built once with a single vectorized Geod.inv.

    The table is not bit-identical to solving at each cell's own longitude: pyproj's solver rounds differently at different absolute longitudes, giving deviations up to ~1e-9 m per hop (relative ~3e-12, and well under a micrometre accumulated over the longest flow path).
    """
    if abs(TRANSFORM.b) > 1e-12 or abs(TRANSFORM.d) > 1e-12:
        raise ValueError(f"_hop_lengths assumes a north-up, axis-aligned transform; got b={TRANSFORM.b}, d={TRANSFORM.d}.")

    rows = np.arange(H, dtype=float)
    a, c, e, f = TRANSFORM.a, TRANSFORM.c, TRANSFORM.e, TRANSFORM.f
    clon = np.full(H, a * 0.5 + c)
    clat = e * (rows + 0.5) + f
    table = np.empty((H, len(NEIGHBOR_CHECKS)))
    for k, (dx, dy, _) in enumerate(NEIGHBOR_CHECKS):
        nlon = np.full(H, a * (dx + 0.5) + c)
        nlat = e * (rows + dy + 0.5) + f
        _, _, seg = _GEOD.inv(clon, clat, nlon, nlat)
        table[:, k] = seg
    table.setflags(write=False)
    return table


def _build_flow_field(direction: np.ndarray, col: int, row: int) -> np.ndarray:
    """Metres-to-outlet for every cell draining to (col, row).

    Expands the upstream frontier one BFS level at a time, whole level vectorized. Every D8 cell drains to exactly one downstream cell, so the "flows into" relation is a forest and the upstream walk from an outlet is a tree: each cell is reached exactly once, by a unique parent. Visit order therefore cannot affect any distance, which is what lets the level expansion replace the per-cell deque loop without changing results.
    """
    dist = np.full((H, W), np.nan)
    dist[row, col] = 0.0
    hop = _hop_lengths()
    fr_r = np.array([row], dtype=np.intp)
    fr_c = np.array([col], dtype=np.intp)

    while fr_r.size:
        base = dist[fr_r, fr_c]
        nxt_r, nxt_c = [], []
        for k, (dx, dy, expected) in enumerate(NEIGHBOR_CHECKS):
            nr, nc = fr_r + dy, fr_c + dx
            on = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            nr, nc = nr[on], nc[on]
            if nr.size == 0:
                continue
            take = (direction[nr, nc] == expected) & np.isnan(dist[nr, nc])
            if not take.any():
                continue
            nr, nc = nr[take], nc[take]
            dist[nr, nc] = base[on][take] + hop[fr_r[on][take], k]
            nxt_r.append(nr)
            nxt_c.append(nc)
        if not nxt_r:
            break
        fr_r = np.concatenate(nxt_r)
        fr_c = np.concatenate(nxt_c)
    return dist


def flow_distance_field_ll(lat: float, lon: float) -> np.ndarray:
    """Per-cell flow distance (m) to the outlet of the sensor at (lat, lon). Cached per rounded
    (lat, lon). Raises ValueError if the sensor is outside the D8 raster extent (Iowa)."""
    key = (round(lat, 6), round(lon, 6))
    if key in _FLOW_FIELD_CACHE:
        _FLOW_FIELD_CACHE.move_to_end(key, last=False)
        return _FLOW_FIELD_CACHE[key]
    direction = load_direction_array()
    col, row = ll_to_image_pixel(lat, lon)
    if not (0 <= col < W and 0 <= row < H):
        raise ValueError(f"sensor ({lat}, {lon}) is outside the D8 raster extent.")
    ocol, orow = _snap_outlet(direction, col, row)
    field = _build_flow_field(direction, ocol, orow)
    _FLOW_FIELD_CACHE[key] = field
    _FLOW_FIELD_CACHE.move_to_end(key, last=False)
    while len(_FLOW_FIELD_CACHE) > _FLOW_FIELD_CACHE_MAX:
        _FLOW_FIELD_CACHE.popitem(last=True)  # oldest use, 14.7 MB back
    return field


def sample_field(field: np.ndarray, col: int, row: int) -> float:
    """Flow distance at a node pixel, recovering basin-divide straddlers via a small search."""
    if 0 <= col < W and 0 <= row < H and np.isfinite(field[row, col]):
        return float(field[row, col])
    R = NODE_SNAP_RADIUS
    c0, c1 = max(0, col - R), min(W, col + R + 1)
    r0, r1 = max(0, row - R), min(H, row + R + 1)
    sub = field[r0:r1, c0:c1]
    fin = np.isfinite(sub)
    if not fin.any():
        return float("nan")
    rs, cs = np.where(fin)
    j = int(np.argmin((cs + c0 - col) ** 2 + (rs + r0 - row) ** 2))
    return float(sub[rs[j], cs[j]])
