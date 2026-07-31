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

import shutil
import sys
from functools import lru_cache
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
#   3: weather_lag_days is indexed BY bucket, one entry per ring, aligned with the weather list
#   4: per-task long-run composition block (features.longrun_from_blocks), year-invariant
ROW_SCHEMA = 4

# Which long-run reductions the reach store CARRIES, as opposed to which ones a model consumes (recipes.LONGRUN_STATS / LIGHT_LONGRUN_STATS). Deliberately the superset: forecast.js resolves columns by name against the booster's own feature list, so an unused packed block costs bytes and nothing else -- while narrowing it here would make every change of modelling mind another 16,760-reach pass. At 3 rings and 9 values that is 108 f32 a reach across both tasks, about +7 MB over the whole store.
LONGRUN_PACK_STATS = ("mean", "sd")


def _features_mod():
    from src.features import features

    return features


@lru_cache(maxsize=1)
def _forecast_years():
    """build_bundle.FORECAST_YEARS, imported lazily -- build_bundle imports THIS module, so a top-level import is a cycle."""
    from widget.static.build_bundle import FORECAST_YEARS

    return tuple(FORECAST_YEARS)


@lru_cache(maxsize=1)
def _longrun_source_years():
    """{block: [year, ...]} actually available to the long-run reduction, per global table.

    The declared window (features.LONGRUN_YEARS) is an upper bound, not what gets reduced: crops stop at 2025 and surplus at 2017. Hashing what is REALLY there is what makes a data refresh invalidate -- a new crop year lands inside the declared window, so the window alone would not move and every cached reach row would keep its stale mean.
    """
    import pandas as pd

    keep = set(_features_mod().LONGRUN_YEARS)
    out = {}
    for block, rel in (("crops", "crops_global.parquet"), ("surplus", "surplus_global.parquet")):
        y = pd.read_parquet(_ROOT / "src" / "data" / "interim" / rel, columns=["year"])["year"]
        out[block] = sorted(int(v) for v in y.unique() if int(v) in keep)
    return out


@lru_cache(maxsize=1)
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

    from src.features import features, recipes

    cfg = {
        "edges": {k: list(v) for k, v in recipes.LIGHT_EDGES.items()},
        "weather_edges": {k: list(v) for k, v in recipes.LIGHT_WEATHER_EDGES.items()},
        "vel": recipes.LIGHT_VEL,
        "lam": recipes.LIGHT_LAM,
        "expT_drop": list(recipes.EXPT_DROP),
        "weather_keep": list(recipes.WEATHER_KEEP),
        "rank": WEATHER_RANK,
        # The long-run block reduces over a WINDOW, so the window is part of what a row means. Both the declared window and the realized per-block year sets go in: a data refresh that adds a crop year inside the training span would shift every reduced value while leaving the declared window untouched, and _is_current would keep serving the old rows.
        "longrun_years": [min(features.LONGRUN_YEARS), max(features.LONGRUN_YEARS)],  # declared window
        "longrun_stats": list(LONGRUN_PACK_STATS),
        "longrun_source_years": _longrun_source_years(),
        # reach_row trims the per-year blocks to these, so a change repacks every reach as NaN for the new year unless it invalidates. Pre-existing gap, closed here.
        "forecast_years": sorted(_forecast_years()),
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

    Every reach here is a single LineString (checked: 16,760 of 16,760), so there is no part indexing.

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
    # snappable; the tombstones drop the reaches NLDI has no basin for -- together, the browser can
    # never return a COMID the reach store has no row for. Rebuild this group if new tombstones
    # appear (build_bundle --only snap_index).
    from widget.static.fetch_basins import tombstoned

    dead = tombstoned()
    fl = fl[(fl["StreamOrde"] >= min_stream_order) & (fl["TotDASqKM"] > 0) & (~fl["COMID"].isin(dead))]
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


def reach_row(comid, sd, outlet_lat, outlet_lon, basis, tasks, years):
    """    Everything the browser needs to score one reach, computed the way training computed it.

    `sd` is a SiteData whose grid carries the basin's cells -- build_reaches.py computes it once and caches it, because the cells are a function of the BASIN and the grid, not of the recipe. Everything this function does to them is recipe-dependent and cheap, which is what makes a retuned bucket edge or decay length a re-aggregation of minutes rather than another D8 pass of hours.

    `tasks` maps a task name to (edges, vel, lam), which come straight from recipes.LIGHT_* so this cannot drift from _light_features. The blocks returned per task are the per-year crop shares and surplus by distance bucket plus the unbucketed exp-decay terms; the static descriptors and the weather projection are shared across tasks.

    Weather is NOT aggregated here. _weather_for_grid predicate-scans a ~270 MB year file per basin, which over thousands of reaches would dominate the build to produce one column. Instead the basin's area weights are projected onto the shared low-rank basis (see build_weather_basis), which reproduces the same area-weighted mean the recipe would have computed to within a few percent of a basin's own variability.

    Returns None when no grid cell intersects the basin -- a real outcome for the smallest reaches, and one the caller records rather than crashes on.
    """
    from src.features import recipes
    from src.features.transformers import bucket_lags

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
        _, cb, cb_exp, sb, sb_exp, lr = recipes._agg_block(
            sd, edges, vel, lam, weather_edges=w_edges, weather=False
        )
        # Reduced inside _agg_block from the UNTRIMMED frames, so it sees the whole record and applies
        # its own window (features.LONGRUN_YEARS) -- not the three forecast years `trim` is about to
        # cut down to. Both stats are kept; pack_chunk takes LONGRUN_PACK_STATS from them.
        out[task] = {
            "longrun": lr,
            "crops": trim(cb).to_dict("records"),
            "crops_expT": trim(cb_exp).to_dict("records"),
            "surplus": trim(sb).to_dict("records"),
            "surplus_expT": trim(sb_exp).to_dict("records"),
        }
        # One lag PER WEATHER BUCKET, keyed exactly as the recipe buckets the block, so the browser shifts each series by what training shifted it by.
        #
        # Indexed BY ring, aligned element-for-element with the weather list above. transformers.bucket_lags returns only the rings that HAVE cells, so its own output is dense -- a basin with cells in b0 and b2 yields two values -- and shipping it that way would leave which ring each lag belongs to to be recovered downstream from the weather list's None pattern. That recovery is exact but needless, and non-contiguous rings are common enough here (roughly one basin in seven) that getting it wrong would be both quiet and routine. An empty ring gets 0, which nothing reads: its weather is NaN.
        lag = bucket_lags(site_data=sd, water_velocity=vel, edges=list(w_edges))
        by_ring = {int(b): int(v) for b, v in lag.items()}
        out[task]["weather_lag_days"] = [by_ring.get(b, 0) for b in range(len(w_edges) + 1)]
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


def longrun_cols(crop_classes, stats=LONGRUN_PACK_STATS):
    """Long-run column names in packed order: per stat, the crop classes by ring, then surplus by ring.

    Generated from the DECLARED geometry, never read off a row -- which is the difference between this and _expT_cols. The exp-decay block is built at edges=(), so every reach carries identical keys and sampling one is safe. This block is bucketed and rings go missing constantly (a basin with no cell inside CLF's 2 km has no b0), so a sampled reach would ship a short name list against the fixed N_BUCKETS stride the packer writes -- misaligning every long-run column for every reach in the bundle, while check_feature_skew still reports clean because the names it does ship all resolve.
    """
    out = []
    for stat in stats:
        out += [f"pct_{cls.lower()}_{stat}_b{b}" for cls in crop_classes for b in range(N_BUCKETS)]
        out += [f"surplus_kgha_norm_{stat}_b{b}" for b in range(N_BUCKETS)]
    return out


def _longrun_values(row, task, crop_classes, stats=LONGRUN_PACK_STATS):
    """One reach's long-run block, flattened in longrun_cols' order. An absent ring is NaN, as everywhere else."""
    lr = row[task].get("longrun") or {}
    return [lr.get(stat, {}).get(c, np.nan) for stat in stats for c in longrun_cols(crop_classes, (stat,))]


def pack_chunk(rows, years, crop_classes, rank):
    """    Pack one spatial chunk of reaches into the layout forecast.js reads.

    [n:i32][rank:i32][n_years:i32][n_buckets:i32][n_crops:i32], then, each of length n: comid i32; the six statics as f32 (lat, lon, basin_area_m2, mean_dist, max_dist, log_basin_area); then per task in (reg, clf) the n*n_buckets lags i8, the n*n_buckets weather offsets f32 and the n*n_buckets*rank weather coefficients f32; then the reg block and the clf block, each n * n_years * (n_crops*n_buckets + n_buckets + n_crops + 1).

    WEATHER IS PER RING AND PER TASK. fuel_moisture_1000h is bucketed by distance exactly as the crop and surplus blocks are -- the light models want fuel_moisture_1000h_b0/_b1/_b2 -- and the two tasks cut their rings at different distances, so each is a different functional of the same shared modes. That costs 2 * n_buckets * (1 + rank) floats a reach instead of 1 + rank, which is most of what a reach weighs and still nothing beside shipping the node-space field.

    An absent ring is NaN, which is what the recipe emits for it and what the booster routes down its default branch -- the same treatment predict() gives a missing distance bucket.

    THE LONG-RUN BLOCK IS APPENDED AT THE TAIL, and its two per-task widths are the FINAL 8 bytes, rather than the header growing to carry them. That is deliberate. Every offset above is computed from a 20-byte header -- forecast.js::decodeChunk hardcodes `o = 20` and check_forecast.py reads the COMID array at a literal offset 20 -- so widening the header would shift the whole body under readers that have no version field to notice with. The parity harness would not crash on that; it would read 8 bytes into the COMID array and compare plausible-looking garbage, taking out the one value-level check exactly when it is needed. At the tail, an old reader decodes everything it knows about correctly and simply never looks at the new bytes.
    """
    rows = sorted(rows, key=lambda r: r["comid"])
    n = len(rows)
    per_task = len(years) * (len(crop_classes) * N_BUCKETS + N_BUCKETS + len(crop_classes) + 1)

    head = np.array([n, rank, len(years), N_BUCKETS, len(crop_classes)], np.int32).tobytes()
    comid = np.array([r["comid"] for r in rows], np.int32).tobytes()
    statics = np.array(
        [[r["lat"], r["lon"], r["basin_area_m2"], r["mean_dist_to_sensor"], r["max_dist_to_sensor"], r["log_basin_area"]]
         for r in rows], np.float32).tobytes()
    lags = b"".join(np.array([r[task]["weather_lag_days"] for r in rows], np.int8).tobytes() for task in ("reg", "clf"))
    offsets = b"".join(
        np.array([[(w or {}).get("offset", np.nan) for w in r[task]["weather"]] for r in rows], np.float32).tobytes()
        for task in ("reg", "clf")
    )
    coefs = b"".join(
        np.array([[(w["coef"] if w else [np.nan] * rank) for w in r[task]["weather"]] for r in rows],
                 np.float32).tobytes()
        for task in ("reg", "clf")
    )
    blocks = b"".join(
        np.array([_row_values(r, task, years, crop_classes) for r in rows], np.float32).tobytes()
        for task in ("reg", "clf")
    )
    lr_names = longrun_cols(crop_classes)
    longrun = b"".join(
        np.array([_longrun_values(r, task, crop_classes) for r in rows], np.float32).tobytes()
        for task in ("reg", "clf")
    )
    # The two widths, at the very END. Both tasks are the same width today, but they are declared per
    # task so a future per-task stats split needs no layout change.
    widths = np.array([len(lr_names), len(lr_names)], np.int32).tobytes()
    assert len(lags) == 2 * n * N_BUCKETS, "packed lags do not match the declared layout"
    assert len(coefs) == 2 * n * N_BUCKETS * rank * 4, "packed weather does not match the declared layout"
    assert len(blocks) == 2 * n * per_task * 4, "packed block size does not match the declared layout"
    # A real raise, not an assert: python -O strips asserts, and a long-run width mismatch misaligns
    # EVERY long-run column for every reach while check_feature_skew still reports clean.
    if len(longrun) != 2 * n * len(lr_names) * 4:
        raise ValueError(
            f"packed long-run block is {len(longrun)} bytes, expected {2 * n * len(lr_names) * 4} "
            f"({n} reaches x 2 tasks x {len(lr_names)} cols x 4)"
        )
    return head + comid + statics + lags + offsets + coefs + blocks + longrun + widths


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


def _expT_cols(row, crop_classes) -> dict:
    """{task: [exp-decay column name, ...]}, in _row_values' packed order.

    The names carry the recipe's lambda (Corn_expT2000), so shipping them lets the browser match the model's exp-decay columns EXACTLY. Matching by prefix instead would bridge a model trained at one lambda onto values computed at another -- the skew deploy.predict._assert_no_skew refuses to score through.
    """
    def names(task):
        cexp, sexp = row[task]["crops_expT"][0], row[task]["surplus_expT"][0]
        out = [next((k for k in cexp if k.startswith(f"{cls}_expT")), f"{cls}_expT") for cls in crop_classes]
        return out + [next((k for k in sexp if k.startswith("surplus_kgha_expT")), "surplus_kgha_expT")]

    return {task: names(task) for task in ("reg", "clf")}


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

    # Clear what a previous pack wrote. A pack emits only the chunks that HAVE current rows, so a
    # leftover file for a chunk this pass has nothing for stays on disk and the browser reads it --
    # under whatever layout it was written with. The manifest would not list it, so export's orphan
    # check catches it before publishing, but every local test until then scores stale data.
    for sub in ("reaches", "basins"):
        shutil.rmtree(out / "forecast" / sub, ignore_errors=True)

    # Stale rows are SKIPPED, not packed. build_reaches leaves a row per COMID whatever schema it was written under, so a pack after a bump would otherwise read an old shape into the new layout -- garbage that looks like data. Counted and reported, since a large number means the pass is behind the recipe.
    fingerprint = recipe_fingerprint()
    by_chunk = {}
    sizes = {}
    stale = 0
    for f in files:
        if not f.stem.isdigit():
            continue
        row = json.loads(f.read_text())
        if row.get("schema") != ROW_SCHEMA or row.get("recipe") != fingerprint:
            stale += 1
            continue
        by_chunk.setdefault(chunk_of(row["lat"], row["lon"]), []).append(row)

        comid = row["comid"]
        src = basin_cache / f"{comid}.json"
        if src.exists():
            basin = gpd.GeoDataFrame.from_features(json.loads(src.read_text())["features"], crs="EPSG:4326")
            rel = f"forecast/basins/{comid}.bin"
            sizes[rel] = write_bin(out / "forecast" / "basins" / f"{comid}.bin", pack_basin(basin))

    if stale:
        print(f"   [skipped] {stale:,} row(s) from an older schema or recipe -- re-run build_reaches to refresh them")
    if not by_chunk:
        raise RuntimeError(
            f"every one of the {len(files):,} reach rows is stale (schema != {ROW_SCHEMA} or a different recipe). "
            "Run `python -m widget.static.build_reaches` before packing."
        )

    for cid, rows in sorted(by_chunk.items()):
        rel = f"forecast/reaches/{cid}.bin"
        sizes[rel] = write_bin(out / "forecast" / "reaches" / f"{cid}.bin", pack_chunk(rows, list(years), crop_classes, rank))
    # weather_cols travels with the chunks so the browser names its weather columns from the RECIPE rather than from a literal in forecast.js -- WEATHER_KEEP is a recipe setting, and renaming it must not need a JS edit. One variable only: the reach row projects a single field onto the shared basis.
    from src.features import recipes

    if len(recipes.WEATHER_KEEP) != 1:
        raise ValueError(f"the reach store projects ONE weather variable; WEATHER_KEEP is {recipes.WEATHER_KEEP}")

    sizes["forecast/reach_chunks.json"] = _write_json_via(
        out / "forecast" / "reach_chunks.json",
        # row_schema/recipe are carried through from the ROWS so build_bundle can tell a stale PACK from a
        # current one. Its skip check is otherwise existence-based, and build_reaches rewrites rows in place
        # under the same names -- so without this stamp a schema bump leaves 17,044 present-but-superseded
        # files and the repack is skipped in silence. See build_bundle._reaches_stale.
        {"bbox": CHUNK_BBOX, "grid": CHUNK_GRID, "chunks": sorted(by_chunk), "n_reaches": len(files),
         "row_schema": ROW_SCHEMA, "recipe": fingerprint,
         "years": list(years), "crop_classes": crop_classes, "n_buckets": N_BUCKETS, "rank": rank,
         "weather_cols": list(recipes.WEATHER_KEEP),
         "expT_cols": _expT_cols(next(iter(by_chunk.values()))[0], crop_classes),
         # Generated from the declared geometry, NOT sampled from a row like expT_cols -- see longrun_cols. Names only: json.dumps writes a bare NaN token that JSON.parse rejects outright, so one value in here would blank the whole forecast.
         "longrun_cols": {task: longrun_cols(crop_classes) for task in ("reg", "clf")}},
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
