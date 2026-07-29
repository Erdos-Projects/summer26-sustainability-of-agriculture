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
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=6, help="how many reaches to check (default 6)")
    p.add_argument("--year", type=int, default=None, help="forecast year (default: the bundle's middle year)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--comids", default=None, help="comma-separated COMIDs to check instead of a sample")
    a = p.parse_args()
    raise SystemExit(main(n=a.n, year=a.year, beta=a.beta, comids=a.comids.split(",") if a.comids else None))
