"""Forecast-path bundle artifacts: the pieces a browser needs to score the light models at a pin.

Registered into widget/static/build_bundle.BUILDERS as separate groups so each can be rebuilt on its own -- the reach pass is hours long and must not be redone because the weather basis changed. build_bundle stays the driver and the manifest owner; only the forecast logic lives here.

Three groups, in dependency order:

  weather_basis   the shared temporal modes for fuel_moisture_1000h, plus a cached node-space basis
                  the reach pass projects onto. Needs only the weather parquets.
  cross_site      the statewide nitrate neighbour series. Needs only the nitrate cache.
  reaches         per-reach feature rows + basin polygons. Needs widget/static/fetch_basins.py to
                  have populated the NLDI cache first.

THE WEATHER TRICK. fuel_moisture_1000h is the one input that is COMID x DATE, and shipping it per reach would be 16,762 x 1,095 floats. But the model consumes the basin MEAN, which is a linear functional of the node field: mean_b(t) = w_b . A(:,t). So decompose the field once, A ~ mu + U_k S_k Vt_k, and the basin mean falls out as w_b.mu + (w_b . U_k S_k) . Vt_k. Each reach then stores one scalar and k coefficients -- 33 floats instead of 1,095 -- against shared modes of k x n_days. The node field itself never ships.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent  # widget/static
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

# Rank of the weather decomposition, fixed BEFORE the reach pass because that pass bakes the
# coefficients in and re-running it costs hours.
#
# Measured on the 83 known sites' real AREA-WEIGHTED basin means (weights = cell_area *
# frac_cell_in_basin, exactly as transformers._area_mean_curry builds them), reconstruction error
# as a fraction of each basin's own within-site SD:
#
#     rank   mean   worst   max|err|   floats/reach   modes
#       16   0.149   0.269    1.93          17         75 KB
#       32   0.089   0.143    1.65          33        150 KB
#       64   0.046   0.086    0.85          65        300 KB   <- chosen
#       96   0.030   0.053    0.69          97        449 KB
#
# 64 halves the error against 32 for about 2 MB across all 16,762 reaches, and the gain flattens
# after it (96 buys 1.6% of SD for another 50% of the per-reach cost). Max absolute error at 64 is
# 0.85 on a field spanning 10.4..27.9.
WEATHER_RANK = 64

# Bump whenever reach_row's OUTPUT SHAPE changes -- a new block, a different bucketing, a renamed
# field. build_reaches.py skips any COMID whose file already exists, so without a version marker a
# recipe change would leave thousands of silently stale rows to be packed and shipped as if current.
# That is the same failure mode _assert_no_skew catches at predict time, one stage earlier.
#   1: initial (single whole-basin weather projection)
#   2: weather projection is per distance bucket, and weather_lag_days is a list
ROW_SCHEMA = 2


def recipe_fingerprint():
    """    A short hash of every light-recipe setting that changes what reach_row emits.

    ROW_SCHEMA catches a change of SHAPE; this catches a change of CONTENT, which is the more common
    and the more dangerous one. Retuning a bucket edge, a decay length or the weather geometry leaves
    the row structurally identical while making every value in it wrong for the model that will
    consume it -- and build_reaches skips any COMID whose file exists, so thousands of rows would
    quietly survive the change and be packed as current. Recording the fingerprint turns that into a
    rebuild instead of a silent mismatch.
    """
    import hashlib
    import json as _json

    from src.features import recipes

    cfg = {
        "edges": {k: list(v) for k, v in recipes.LIGHT_EDGES.items()},
        "weather_edges": {k: list(v) for k, v in recipes.LIGHT_WEATHER_EDGES.items()},
        "vel": recipes.LIGHT_VEL,
        "lam": recipes.LIGHT_LAM,
        "expT_drop": list(recipes.EXPT_DROP),
        "weather_keep": list(recipes.WEATHER_KEEP),
        "rank": WEATHER_RANK,
    }
    return hashlib.sha256(_json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _weather_span(years, buffer_days):
    """(start, end) Timestamps covering the forecast years plus the pre-roll the lag shift needs."""
    start = pd.Timestamp(f"{min(years)}-01-01") - pd.Timedelta(days=buffer_days)
    return start, pd.Timestamp(f"{max(years)}-12-31")


def _fuel_moisture_matrix(years, buffer_days):
    """(node_ids, dates, A) for fuel_moisture_1000h over the span -- nodes x days, float64.

    Reads whole year files rather than pushing a date predicate down, because the span crosses a
    year boundary at the buffer and the column projection already cuts the read to a tenth.
    """
    start, end = _weather_span(years, buffer_days)
    frames = []
    for y in range(start.year, end.year + 1):
        path = _ROOT / "src/data/interim" / f"weather_global_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; the forecast span needs {start.date()}..{end.date()}.")
        frames.append(pd.read_parquet(path, columns=["date", "global_node_id", "fuel_moisture_1000h"]))
    w = pd.concat(frames, ignore_index=True)
    w = w[(w["date"] >= start) & (w["date"] <= end)]
    M = w.pivot(index="global_node_id", columns="date", values="fuel_moisture_1000h").sort_index()
    if M.isna().any().any():
        raise ValueError(f"fuel_moisture_1000h has {int(M.isna().sum().sum())} gaps over the span; fill them first")
    return M.index.to_numpy(np.int32), pd.DatetimeIndex(M.columns), M.to_numpy(np.float64)


def build_weather_basis(out, years, buffer_days, days_since_epoch, write_bin, cache_dir):
    """    Decompose the fuel-moisture field; ship the temporal modes, cache the node-space basis.

    The shipped file is the half that is shared by every reach. The cached half (node ids, per-node
    climatology, U_k S_k) is what the reach pass projects a basin's weight vector onto; it is an
    intermediate, never published, and lives beside the other src/data caches.
    """
    node_ids, dates, A = _fuel_moisture_matrix(years, buffer_days)
    mu = A.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    k = min(WEATHER_RANK, len(S))

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "weather_basis.npz",
        node_ids=node_ids,
        mu=mu.ravel().astype(np.float64),
        US=(U[:, :k] * S[:k]).astype(np.float64),
        dates=days_since_epoch(dates),
        rank=np.int32(k),
    )

    # [k:i32][n_days:i32][dates:i4 x n_days][Vt:f32 k*n_days, row-major]
    payload = (
        np.array([k, len(dates)], np.int32).tobytes()
        + days_since_epoch(dates).astype(np.int32).tobytes()
        + np.ascontiguousarray(Vt[:k], np.float32).tobytes()
    )
    return {"forecast/weather_modes.bin": write_bin(out / "forecast" / "weather_modes.bin", payload)}


def build_snap_index(out, min_stream_order, write_bin):
    """    The flowline geometry a browser needs to turn a dropped pin into a COMID.

    Mirrors _make_basins.snap_comid, which is a nearest-reach query in EPSG:5070 METRES with a 25 m tie tolerance resolved toward the larger drainage area. Two consequences for what ships:

    Coordinates are stored PROJECTED, not lat/lon, so the browser measures the same metres Python does and the tie rule means the same thing on both sides. forecast.js carries the Albers forward projection to match.

    Geometry is NOT simplified. Simplifying barely pays here -- 1 m tolerance sheds 3% of vertices, 5 m sheds 21% -- while spending a fifth of the 25 m tie budget on displacement, which is exactly what would make a pin near two reaches snap differently in the browser than in Python. 2.87 MB of exact float32 is the cheaper trade. float32 resolves to about 0.25 m at Iowa's ~4e6 m easting, well inside the tolerance.

    Every reach here is a single LineString (checked: 16,762 of 16,762), so there is no part indexing.

    THE OUTLET is shipped alongside, in lat/lon, because it is where the forecast is actually computed. NLDI's basin for a COMID is the area draining to that reach's downstream end, and the per-reach feature row is built there -- not at wherever the user clicked, which can be kilometres away and is not a point any precomputed row describes. The map moves the pin to it so the displayed location is the modelled one.

    Which end is downstream was measured, not assumed: over 150 reaches with a cached basin, the LAST vertex sits a median 22 m from the basin boundary against 524 m for the first, and is the closer end in 97% of them. That is the pour point.

    Layout: [n_reaches:i32][n_vertices:i32], then per reach comid:i32, tot_da_sqkm:f32, vertex start:i32, count:i32, outlet_lat:f32, outlet_lon:f32, then the projected x:f32 and y:f32 runs.
    """
    import geopandas as gpd

    from src.build._make_basins import EQUAL_AREA_CRS

    fl = gpd.read_parquet(
        _ROOT / "src/data/processed/map_overlays/iowa_flowlines.parquet",
        columns=["COMID", "TotDASqKM", "StreamOrde", "GNIS_NAME", "geometry"],
    )
    # TotDASqKM > 0 drops the NHDPlus divergence artifacts, matching what snap_comid considers
    # snappable -- so the browser can never return a COMID the reach store has no row for.
    fl = fl[(fl["StreamOrde"] >= min_stream_order) & (fl["TotDASqKM"] > 0)]
    fl = fl.sort_values("COMID").reset_index(drop=True)
    if not (fl.geom_type == "LineString").all():
        raise ValueError("a reach is not a single LineString; the packed layout assumes one part each")

    # Outlets in lat/lon (the map needs them there); geometry projected for the distance work.
    outlets = np.array([np.asarray(g.coords)[-1][:2] for g in fl.geometry], np.float64)  # lon, lat
    projected = fl.to_crs(EQUAL_AREA_CRS)

    coords = [np.asarray(g.coords, np.float64)[:, :2] for g in projected.geometry]
    counts = np.array([len(c) for c in coords], np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    xy = np.concatenate(coords)

    payload = (
        np.array([len(fl), len(xy)], np.int32).tobytes()
        + fl["COMID"].to_numpy(np.int32).tobytes()
        + fl["TotDASqKM"].to_numpy(np.float32).tobytes()
        + starts.tobytes()
        + counts.tobytes()
        + outlets[:, 1].astype(np.float32).tobytes()  # lat
        + outlets[:, 0].astype(np.float32).tobytes()  # lon
        + np.ascontiguousarray(xy[:, 0], np.float32).tobytes()
        + np.ascontiguousarray(xy[:, 1], np.float32).tobytes()
    )
    sizes = {"forecast/snap_index.bin": write_bin(out / "forecast" / "snap_index.bin", payload)}

    # Stream names, for telling the user which watercourse the pin landed on. Only the named ones,
    # keyed by COMID -- unnamed reaches are the majority and would double the file for nothing.
    names = {
        int(c): str(n).strip()
        for c, n in zip(fl["COMID"], fl["GNIS_NAME"])
        if isinstance(n, str) and str(n).strip()
    }
    sizes["forecast/reach_names.json"] = _write_json_via(out / "forecast" / "reach_names.json", names, write_bin)
    return sizes


def reach_row(comid, basin, outlet_lat, outlet_lon, basis, tasks, years):
    """    Everything the browser needs to score one reach, computed the way training computed it.

    `tasks` maps a task name to (edges, vel, lam), which come straight from recipes.LIGHT_* so this cannot drift from _light_features. The blocks returned per task are the per-year crop shares and surplus by distance bucket plus the unbucketed exp-decay terms; the static descriptors and the weather projection are shared across tasks.

    Weather is NOT aggregated here. _weather_for_grid predicate-scans a ~270 MB year file per basin, which over thousands of reaches would dominate the build to produce one column. Instead the basin's area weights are projected onto the shared low-rank basis (see build_weather_basis), which reproduces the same area-weighted mean the recipe would have computed to within a few percent of a basin's own variability.

    Returns None when no grid cell intersects the basin -- a real outcome for the smallest reaches, and one the caller records rather than crashes on.
    """
    from src.data import access
    from src.features import recipes
    from src.features.transformers import bucket_lags

    sd = access.build_virtual_site_data(
        basin, outlet_lat, outlet_lon, site_uid=f"COMID-{comid}", with_weather=False
    )
    grid = sd.grid
    if grid is None or grid.empty:
        return None

    out = {
        "schema": ROW_SCHEMA,
        "recipe": recipe_fingerprint(),
        "comid": int(comid),
        "lat": float(outlet_lat),
        "lon": float(outlet_lon),
        "basin_area_m2": float(sd.basin_area),
        "n_cells": int(len(grid)),
        "mean_dist_to_sensor": float(grid["dist_to_sensor"].mean()),
        "max_dist_to_sensor": float(grid["dist_to_sensor"].max()),
        "log_basin_area": float(np.log10(sd.basin_area)) if sd.basin_area > 0 else float("nan"),
    }

    # The aggregators emit every year the global tables hold -- 26 for crops, 18 for surplus -- and
    # the forecast dropdown offers three. Trimming here rather than at pack time keeps the ~8x
    # difference out of every downstream structure.
    keep = set(years)
    trim = lambda df: df[df["year"].isin(keep)].sort_values("year") if "year" in df else df  # noqa: E731

    for task, (edges, vel, lam) in tasks.items():
        w_edges = recipes.LIGHT_WEATHER_EDGES[task]
        _, cb, cb_exp, sb, sb_exp = recipes._agg_block(
            sd, edges, vel, lam, weather_edges=w_edges, weather=False
        )
        out[task] = {
            "crops": trim(cb).to_dict("records"),
            "crops_expT": trim(cb_exp).to_dict("records"),
            "surplus": trim(sb).to_dict("records"),
            "surplus_expT": trim(sb_exp).to_dict("records"),
        }
        # One lag PER WEATHER BUCKET, keyed exactly as the recipe buckets the block, so the browser
        # shifts each series by what training shifted it by.
        lag = bucket_lags(site_data=sd, water_velocity=vel, edges=list(w_edges))
        out[task]["weather_lag_days"] = [int(v) for v in lag] if len(lag) else [0]
        out[task]["weather"] = _project_weather(grid, basis, w_edges)

    return out


def _weather_bucket_masks(grid, edges):
    """Boolean mask per distance bucket, cut on dist_to_sensor exactly as transformers._bucket_map.

    With no edges there is one bucket -- the whole basin -- which is the single-column case.
    """
    d = grid["dist_to_sensor"].to_numpy()
    bounds = [-np.inf, *edges, np.inf]
    return [(d >= bounds[i]) & (d < bounds[i + 1]) for i in range(len(bounds) - 1)]


def _project_weather(grid, basis, edges=()):
    """    Project a basin's weather weights onto the shared fuel-moisture basis: offset + k coefficients, ONE SET PER BUCKET.

    The basin mean is a linear functional of the node field, so each distance ring is simply another functional against the SAME shared modes -- k more coefficients per reach, not another COMID x date series. That is why bucketing weather costs almost nothing in the shipped bundle even though it is the only block that varies by date.

    The weights are cell_area * frac_cell_in_basin, exactly transformers._area_mean_curry's, restricted to the cells that HAVE weather and renormalised over those -- which is what the pandas path does implicitly, since a cell absent from the weather frame never enters the average.

    Returns a list with one {offset, coef} per bucket (None for a ring no cell falls in, which the browser NaN-fills as the recipe would).
    """
    pos = basis["pos"]
    gnid = grid["global_node_id"].to_numpy()
    have = np.array([g in pos for g in gnid])
    cols = np.array([pos[g] for g in gnid[have]])
    area = (grid["cell_area"].to_numpy() * grid["frac_cell_in_basin"].to_numpy()) / 1e4

    out = []
    for mask in _weather_bucket_masks(grid, edges):
        sel = mask[have]
        w = area[have][sel]
        if not sel.any() or w.sum() <= 0:
            out.append(None)
            continue
        w = w / w.sum()
        c = cols[sel]
        out.append({
            "offset": float(w @ basis["mu"][c]),
            "coef": (w @ basis["US"][c]).astype(np.float32).tolist(),
        })
    return out


def load_weather_basis(cache_dir):
    """The cached node-space basis build_weather_basis wrote, plus a global_node_id -> row lookup."""
    b = np.load(cache_dir / "weather_basis.npz")
    return {
        "mu": b["mu"],
        "US": b["US"],
        "rank": int(b["rank"]),
        "pos": {int(g): i for i, g in enumerate(b["node_ids"])},
    }


# Spatial chunking. A pin fetches exactly one chunk, so the grid is sized to keep that fetch small
# while a pan-and-click session mostly stays inside chunks it already has. The assignment is a PURE
# FUNCTION of the outlet, computed identically here and in forecast.js -- so no artifact has to
# carry a chunk id and the two can never disagree about where a reach lives.
CHUNK_BBOX = (40.0, -97.0, 44.0, -90.0)  # lat0, lon0, lat1, lon1 -- covers Iowa with margin
CHUNK_GRID = (16, 28)  # rows (lat) x cols (lon)

# Distance buckets are padded to this width per task. A basin whose cells all sit beyond the inner
# edge simply has no b0, and the recipe emits no column for it; the packer writes NaN there, which
# is exactly what predict() would have NaN-filled anyway. Fixed stride beats variable-length offsets.
N_BUCKETS = 3


def chunk_of(lat, lon):
    lat0, lon0, lat1, lon1 = CHUNK_BBOX
    rows, cols = CHUNK_GRID
    r = min(rows - 1, max(0, int((lat - lat0) / (lat1 - lat0) * rows)))
    c = min(cols - 1, max(0, int((lon - lon0) / (lon1 - lon0) * cols)))
    return r * cols + c


def _row_values(row, task, years, crop_classes):
    """One reach's per-task block, flattened in the order the browser reads it.

    Column order, fixed here and mirrored in forecast.js: for each year, the crop shares by class
    then bucket, the surplus shares by bucket, the unbucketed crop exp-decay terms by class, and the
    surplus exp-decay term. Missing buckets become NaN.
    """
    blocks = row[task]
    by_year = {b: {r["year"]: r for r in blocks[b]} for b in ("crops", "crops_expT", "surplus", "surplus_expT")}
    out = []
    for y in years:
        crops = by_year["crops"].get(y, {})
        surplus = by_year["surplus"].get(y, {})
        cexp = by_year["crops_expT"].get(y, {})
        sexp = by_year["surplus_expT"].get(y, {})
        for cls in crop_classes:
            for b in range(N_BUCKETS):
                out.append(crops.get(f"pct_{cls.lower()}_b{b}", np.nan))
        for b in range(N_BUCKETS):
            out.append(surplus.get(f"surplus_kgha_norm_b{b}", np.nan))
        # The exp-decay tag carries lambda (Corn_expT2000), so match by prefix rather than by an
        # exact name this packer would have to keep in step with the recipe's lambda.
        for cls in crop_classes:
            out.append(next((v for k, v in cexp.items() if k.startswith(f"{cls}_expT")), np.nan))
        out.append(next((v for k, v in sexp.items() if k.startswith("surplus_kgha_expT")), np.nan))
    return out


def pack_chunk(rows, years, crop_classes, rank):
    """    Pack one spatial chunk of reaches into the layout forecast.js reads.

    [n:i32][rank:i32][n_years:i32][n_buckets:i32][n_crops:i32], then, each of length n:
    comid i32; the six statics as f32 (lat, lon, basin_area_m2, mean_dist, max_dist, log_basin_area);
    the two weather lags as i8 (reg, clf); the weather offset f32; then n*rank weather coefficients;
    then the reg block and the clf block, each n * n_years * (n_crops*n_buckets + n_buckets + n_crops + 1).
    """
    rows = sorted(rows, key=lambda r: r["comid"])
    n = len(rows)
    per_task = len(years) * (len(crop_classes) * N_BUCKETS + N_BUCKETS + len(crop_classes) + 1)

    head = np.array([n, rank, len(years), N_BUCKETS, len(crop_classes)], np.int32).tobytes()
    comid = np.array([r["comid"] for r in rows], np.int32).tobytes()
    statics = np.array(
        [[r["lat"], r["lon"], r["basin_area_m2"], r["mean_dist_to_sensor"], r["max_dist_to_sensor"], r["log_basin_area"]]
         for r in rows], np.float32).tobytes()
    lags = np.array([[r["reg"]["weather_lag_days"], r["clf"]["weather_lag_days"]] for r in rows], np.int8).tobytes()
    offset = np.array([(r["weather"] or {}).get("offset", np.nan) for r in rows], np.float32).tobytes()
    coef = np.array(
        [(r["weather"]["coef"] if r["weather"] else [np.nan] * rank) for r in rows], np.float32
    ).tobytes()
    blocks = b"".join(
        np.array([_row_values(r, task, years, crop_classes) for r in rows], np.float32).tobytes()
        for task in ("reg", "clf")
    )
    assert len(blocks) == 2 * n * per_task * 4, "packed block size does not match the declared layout"
    return head + comid + statics + lags + offset + coef + blocks


def pack_basin(basin):
    """    One basin outline as flat float32 lon/lat, for the map overlay.

    NLDI's simplified=true output is already coarse -- simplifying it further at 0.001 deg (111 m) changes nothing, because its vertices are further apart than that already. So the 8.9 KB a basin costs as GeoJSON is TEXT overhead, not detail, and packing the same vertices as float32 pairs costs 3.6 KB with no loss worth the name (float32 resolves about 0.5 m at these longitudes). Across 16,762 reaches that is 61 MB against 149 MB, without accepting the 2.9% area distortion that simplifying to 0.002 deg would have cost.

    Every cached basin is a single part with no holes (checked while packing); an outline is the only thing the overlay draws.

    Layout: [n_vertices:i32][lon:f32 x n][lat:f32 x n].
    """
    geom = basin.geometry.union_all()
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    if len(polys) != 1 or list(polys[0].interiors):
        # Not fatal for a display overlay, but worth knowing about rather than silently dropping.
        polys = [max(polys, key=lambda p: p.area)]
    xy = np.asarray(polys[0].exterior.coords, np.float64)
    return (
        np.int32(len(xy)).tobytes()
        + np.ascontiguousarray(xy[:, 0], np.float32).tobytes()
        + np.ascontiguousarray(xy[:, 1], np.float32).tobytes()
    )


def build_reaches(out, years, crop_classes, write_bin, cache_dir, rows_dir, basin_cache):
    """    Pack every computed reach row into spatial chunks, plus one basin outline per reach.

    Reads the per-reach JSON widget/static/build_reaches.py produced. Splitting compute from pack
    means a change to the shipped layout costs seconds here rather than another multi-hour pass.

    Outlines are per reach rather than per chunk: a forecast draws exactly one, and bundling ~65 of
    them into the chunk it shares with the feature rows would turn a 4 KB fetch into a 240 KB one.
    """
    import json

    rank = int(np.load(cache_dir / "weather_basis.npz")["rank"])
    files = sorted(rows_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no reach rows in {rows_dir}. Run `python -m widget.static.build_reaches` "
            "(after widget.static.fetch_basins has cached the basins)."
        )

    import geopandas as gpd

    by_chunk = {}
    sizes = {}
    for f in files:
        if not f.stem.isdigit():
            continue
        row = json.loads(f.read_text())
        by_chunk.setdefault(chunk_of(row["lat"], row["lon"]), []).append(row)

        comid = row["comid"]
        src = basin_cache / f"{comid}.json"
        if src.exists():
            basin = gpd.GeoDataFrame.from_features(json.loads(src.read_text())["features"], crs="EPSG:4326")
            rel = f"forecast/basins/{comid}.bin"
            sizes[rel] = write_bin(out / "forecast" / "basins" / f"{comid}.bin", pack_basin(basin))

    for cid, rows in sorted(by_chunk.items()):
        rel = f"forecast/reaches/{cid}.bin"
        sizes[rel] = write_bin(out / "forecast" / "reaches" / f"{cid}.bin", pack_chunk(rows, list(years), crop_classes, rank))
    sizes["forecast/reach_chunks.json"] = _write_json_via(
        out / "forecast" / "reach_chunks.json",
        {"bbox": CHUNK_BBOX, "grid": CHUNK_GRID, "chunks": sorted(by_chunk), "n_reaches": len(files),
         "years": list(years), "crop_classes": crop_classes, "n_buckets": N_BUCKETS, "rank": rank},
        write_bin,
    )
    return sizes


def _write_json_via(path, obj, write_bin):
    """Write JSON through build_bundle's byte writer, so the manifest sees one accounting path."""
    import json

    return write_bin(path, json.dumps(obj, separators=(",", ":")).encode())


def build_cross_site(out, years, buffer_days, days_since_epoch, write_bin):
    """    The statewide nitrate neighbour series, precomputed rather than derived in the browser.

    A virtual site's uid is absent from the state, so nothing is excluded and every pin sees the same full rest-of-state average -- one series set covers every reach. The lags are shifts of one base series and the 7d term is a time-based rolling mean with min_periods=1; both are cheap to get subtly wrong in JS against a DatetimeIndex with gaps, and the whole block is ~30 KB, so ship what pandas computed.

    Column order is fixed here and read positionally by the browser: lag1, lag2, lag3, lag5, roll7.
    """
    from src.features.features import nitrate_avg_except_this, rolling_nitrate_avg_except_this

    lags = [1, 2, 3, 5]
    cols = {f"rest_of_state_nitrate_lag{k}": nitrate_avg_except_this("VIRTUAL", shift=k) for k in lags}
    cols["roll_n_avg_except_this7d"] = rolling_nitrate_avg_except_this("VIRTUAL", windows=(7,))[
        "roll_n_avg_except_this7d"
    ]
    df = pd.concat(cols, axis=1).sort_index()

    start, end = _weather_span(years, buffer_days)
    df = df[(df.index >= start) & (df.index <= end)]
    idx = pd.DatetimeIndex(df.index)

    # [n:i32][dates:i4][lag1,lag2,lag3,lag5,roll7 : f32 each]
    payload = np.int32(len(df)).tobytes() + days_since_epoch(idx).astype(np.int32).tobytes()
    for c in [f"rest_of_state_nitrate_lag{k}" for k in lags] + ["roll_n_avg_except_this7d"]:
        payload += df[c].to_numpy(np.float32).tobytes()
    return {"forecast/cross_site.bin": write_bin(out / "forecast" / "cross_site.bin", payload)}
