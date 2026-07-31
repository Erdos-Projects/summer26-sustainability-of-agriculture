"""Does the browser's forecast agree with the Python one? Same reach, same basin, both paths run, curves compared.

    python -m widget.static.check_forecast              # 6 reaches, the bundle's middle forecast year
    python -m widget.static.check_forecast --n 12 --year 2016
    python -m widget.static.check_forecast --comids 6594448,2164507

BOTH SIDES ARE GIVEN THE SAME BASIN -- the cached NLDI polygon the reach row was built from, at that reach's outlet. Neither side snaps. The pin location is a model INPUT, not a label: dist_to_sensor is measured to it and the light recipes cut their rings at 2/5/50 km, so scoring the same reach from its midpoint rather than its outlet moves features across ring boundaries and shifts the prediction by ~0.15-0.36 mg/L -- comparable to everything this harness is trying to measure. Snapping is checked separately (_make_basins.snap_comid vs forecast.js::snapComid).

What is left is the real difference: Python aggregates the cells through recipes._agg_block and reads the weather grid; the browser reads a precomputed row and reconstructs weather from the rank-64 basis. EXACT AGREEMENT IS NOT EXPECTED, and the tolerance is not a knob to widen until it passes: only the weather term is approximated (WEATHER_RANK, a measured mean 0.046 of a basin's within-site SD), so a disagreement well above that scale is structural -- a stale row, a mispacked chunk, a renamed column.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent  # widget/static
_WIDGET = _HERE.parent
_ROOT = _WIDGET.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_WIDGET))  # the widget uses flat imports

RUNNER = _HERE / "_forecast_parity.cjs"
_EPOCH = pd.Timestamp("1970-01-01")


def _packed_comids() -> list[int]:
    """COMIDs the SHIPPED bundle can actually forecast, read from the packed chunks.

    Sampled from the bundle rather than from src/data/cache/reach_rows, so a reach that was computed but never packed is not silently treated as available -- that gap is one of the things worth catching.
    """
    import struct

    out = []
    for p in sorted((_WIDGET / "assets" / "data" / "forecast" / "reaches").glob("*.bin")):
        buf = p.read_bytes()
        n = struct.unpack_from("<i", buf, 0)[0]
        out.extend(struct.unpack_from(f"<{n}i", buf, 20))
    return sorted(out)


def _outlets(comids) -> dict:
    """COMID -> (lat, lon) of the reach's downstream end -- where its row was built, so where both sides score."""
    import geopandas as gpd

    fl = gpd.read_parquet(
        _ROOT / "src/data/processed/map_overlays/iowa_flowlines.parquet", columns=["COMID", "geometry"]
    )
    fl = fl[fl["COMID"].isin(set(comids))]
    return {int(c): (float(np.asarray(g.coords)[-1][1]), float(np.asarray(g.coords)[-1][0]))
            for c, g in zip(fl["COMID"], fl.geometry)}


def _site_data(comid: int, lat: float, lon: float, year: int):
    """SiteData for one reach from its CACHED NLDI basin -- what build_reaches fed reach_row, plus weather.

    build_virtual_basin is deliberately not used: it re-snaps the pin and re-fetches from NLDI, which is the whole thing this harness must hold fixed.
    """
    import geopandas as gpd

    from src.data import access
    from widget.static.build_reaches import _repair
    from widget.static.fetch_basins import CACHE as BASIN_CACHE

    path = BASIN_CACHE / f"{comid}.json"
    if not path.exists():
        raise FileNotFoundError(f"no cached NLDI basin for COMID {comid}")
    basin = _repair(gpd.GeoDataFrame.from_features(json.loads(path.read_text())["features"], crs="EPSG:4326"))
    pad = pd.Timedelta(days=62)  # matches build_virtual_basin's window, so the rolling terms have lead-in
    return access.build_virtual_site_data(
        basin, lat, lon, site_uid=f"COMID-{comid}",
        weather_start=pd.Timestamp(f"{year}-01-01") - pad, weather_end=pd.Timestamp(f"{year}-12-31") + pad,
    )


def _days(idx) -> np.ndarray:
    return ((pd.DatetimeIndex(idx) - _EPOCH) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)


def _compare(py, js) -> dict:
    """Align the two curves on DATE and report the spread. Dates are compared, not assumed: a spine built from the weather calendar and one built from the recipe's target_year_spine can differ in length, and averaging over a misaligned pair would hide it."""
    common = np.intersect1d(_days(py.reg.index), np.asarray(js["dates"], dtype=np.int64))
    pi = pd.Series(py.reg.to_numpy(), index=_days(py.reg.index)).reindex(common).to_numpy()
    pc = pd.Series(py.clf.to_numpy(), index=_days(py.clf.index)).reindex(common).to_numpy()
    ji = pd.Series(js["reg"], index=np.asarray(js["dates"], dtype=np.int64)).reindex(common).to_numpy()
    jc = pd.Series(js["clf"], index=np.asarray(js["dates"], dtype=np.int64)).reindex(common).to_numpy()
    return {
        "n_py": len(py.reg), "n_js": len(js["dates"]), "n_common": len(common),
        "reg_max": float(np.nanmax(np.abs(pi - ji))) if len(common) else float("nan"),
        "reg_mean": float(np.nanmean(np.abs(pi - ji))) if len(common) else float("nan"),
        "reg_scale": float(np.nanmean(pi)) if len(common) else float("nan"),
        "clf_max": float(np.nanmax(np.abs(pc - jc))) if len(common) else float("nan"),
        "clf_mean": float(np.nanmean(np.abs(pc - jc))) if len(common) else float("nan"),
        "tau_py": py.tau, "tau_js": (js.get("op") or {}).get("tau"),
    }


def check_longrun(comid, lat, lon, year) -> dict:
    """Packed long-run block vs the recipe's own, column by column, for one reach. {} when sound.

    The curve comparison below cannot do this job. A boosted model is often near-insensitive to a handful of its ~80 columns, so a wholesale long-run misalignment can land inside a tolerance sized for the rank-64 weather approximation -- and if the block were packed all-NaN, the two sides would agree trivially, NaN against NaN. This reads the actual floats out of the shipped chunk and demands they equal what recipes builds, EXACTLY: unlike weather there is no approximation here to excuse a difference, so anything but bit-equality after a float32 round-trip is a packing or ordering bug.
    """
    import struct

    from src.features import recipes
    from widget.static.build_forecast import N_BUCKETS, longrun_cols

    chunk_dir = _WIDGET / "assets" / "data" / "forecast" / "reaches"
    meta = json.loads((chunk_dir.parent / "reach_chunks.json").read_text())
    names = meta.get("longrun_cols")
    if not names:
        return {"error": "reach_chunks.json carries no longrun_cols; repack with build_bundle --only reaches"}

    for p in sorted(chunk_dir.glob("*.bin")):
        buf = p.read_bytes()
        n, rank, n_years, n_buckets, n_crops = struct.unpack_from("<5i", buf, 0)
        ids = list(struct.unpack_from(f"<{n}i", buf, 20))
        if comid not in ids:
            continue
        i = ids.index(comid)
        w_reg, w_clf = struct.unpack_from("<2i", buf, len(buf) - 8)
        per_year = n_crops * n_buckets + n_buckets + n_crops + 1
        end = len(buf) - 8
        base = end - 4 * n * (w_reg + w_clf)
        out = {}
        for task, width, off in (("reg", w_reg, base), ("clf", w_clf, base + 4 * n * w_reg)):
            cols = names[task]
            if len(cols) != width:
                out[task] = f"manifest lists {len(cols)} columns, chunk packs {width}"
                continue
            packed = dict(zip(cols, struct.unpack_from(f"<{width}f", buf, off + 4 * i * width)))
            sd = _site_data(comid, lat, lon, year)
            # An explicit spine: a virtual site has no water, so the default (daily_nitrate's index) has
            # nothing to build from. The long-run columns are site-constant, so which dates these are
            # does not matter -- only that the frame builds.
            spine = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            frame = recipes.build_feature_frame(sd, task=task, spine=spine, light=True)
            bad = []
            for c, pv in packed.items():
                if c not in frame.columns:
                    continue  # not a column this recipe emits (an absent ring, or a stat it does not take)
                fv = np.float32(frame[c].iloc[0])
                if np.isnan(fv) and np.isnan(pv):
                    continue
                if not (np.isnan(fv) or np.isnan(pv)) and np.float32(pv) == fv:
                    continue
                bad.append(f"{c}: packed {pv!r} vs recipe {float(fv)!r}")
            checked = sum(1 for c in packed if c in frame.columns)
            if bad:
                out[task] = f"{len(bad)}/{checked} columns differ: " + "; ".join(bad[:4])
            elif not checked:
                out[task] = f"none of the {width} packed columns appear in the recipe frame"
        return out
    return {"error": f"COMID {comid} is not in any packed chunk"}


def main(n=6, year=None, beta=2.0, comids=None) -> int:
    import bundle
    import model_interface

    years = bundle.forecast_years()
    year = int(year) if year else years[len(years) // 2]
    if year not in years:
        raise SystemExit(f"year {year} is not in the bundle ({years})")

    available = _packed_comids()
    if not available:
        raise SystemExit(
            "no packed reach chunks in the bundle. Run `python -m widget.static.build_bundle --only reaches`."
        )
    if comids:
        picked = [int(c) for c in comids]
        if missing := [c for c in picked if c not in set(available)]:
            raise SystemExit(f"not in the packed set: {missing}")
    else:
        # Evenly through the COMID order rather than at random: COMIDs run roughly geographically, so
        # a stride samples the state instead of one corner of it, and the run is reproducible.
        step = max(1, len(available) // n)
        picked = available[::step][:n]

    outlets = _outlets(picked)
    cases = [{"comid": c, "lat": outlets[c][0], "lon": outlets[c][1], "year": year, "beta": beta}
             for c in picked if c in outlets]
    print(f"{len(cases)} reach(es), year {year}, beta {beta}  ({len(available):,} reaches packed)\n")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cases, fh)
        cases_path = fh.name
    proc = subprocess.run(["node", str(RUNNER), cases_path], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("the browser-side runner failed")
    js_results = json.loads(proc.stdout)

    print(f"{'COMID':>10} {'rows':>10} {'reg |Δ| max':>12} {'mean':>8} {'of mean':>9} "
          f"{'clf |Δ| max':>12} {'mean':>8}")
    rows = []
    for case, js in zip(cases, js_results):
        comid = case["comid"]
        if "error" in js:
            print(f"{comid:>10}  browser: {js['error']}")
            continue
        py = model_interface.forecast_site_data(_site_data(comid, case["lat"], case["lon"], year), year, beta=beta)
        c = _compare(py, js)
        rows.append(c)
        pct = 100 * c["reg_mean"] / abs(c["reg_scale"]) if c["reg_scale"] else float("nan")
        print(f"{comid:>10} {c['n_common']:>5}/{c['n_py']:<4} "
              f"{c['reg_max']:>12.4f} {c['reg_mean']:>8.4f} {pct:>8.2f}% {c['clf_max']:>12.4f} {c['clf_mean']:>8.4f}")

    if rows:
        print(f"\nover {len(rows)} reach(es):  reg |Δ| worst {max(r['reg_max'] for r in rows):.4f} mg/L, "
              f"mean {np.mean([r['reg_mean'] for r in rows]):.4f}  |  "
              f"clf |Δ| worst {max(r['clf_max'] for r in rows):.4f}, "
              f"mean {np.mean([r['clf_mean'] for r in rows]):.4f}")
        taus = {(r["tau_py"], r["tau_js"]) for r in rows}
        print(f"  operating point tau (python, browser): {sorted(taus)}")

    # Run on the FIRST case only: it re-derives the recipe frame per task, and one reach is enough
    # because a packing or ordering fault is global -- the layout is shared by every chunk.
    if cases:
        c0 = cases[0]
        lr = check_longrun(c0["comid"], c0["lat"], c0["lon"], year)
        if lr:
            print(f"\n  long-run block ({c0['comid']}): MISMATCH")
            for k, v in lr.items():
                print(f"    {k}: {v}")
            return 1
        print(f"\n  long-run block ({c0['comid']}): exact match against the recipe, both tasks")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=6, help="how many reaches to check (default 6)")
    p.add_argument("--year", type=int, default=None, help="forecast year (default: the bundle's middle year)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--comids", default=None, help="comma-separated COMIDs to check instead of a sample")
    a = p.parse_args()
    raise SystemExit(main(n=a.n, year=a.year, beta=a.beta, comids=a.comids.split(",") if a.comids else None))
