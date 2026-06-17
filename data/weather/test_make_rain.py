"""Single-site smoke test for make_rain.py.

Pulls 1 year of IEM rain data from a site's first observation date,
applies the basin filter, and displays the result several ways.

Usage:
    python test_make_rain.py [site_uid]

Default site: USGS-05412500 (Maquoketa River near Maquoketa, IA)
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── import helpers from the sibling make_rain module ─────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from make_rain import (
    _BASIN_DIR,
    _STATS_FILE,
    _RAW_DIR,
    _date_range,
    _download_shapefile,
    _parse_day_gdf,
    _basin_filter,
)

DEFAULT_UID = "USGS-05482500"


def run_test(site_uid: str) -> pd.DataFrame | None:
    # ── resolve date range ────────────────────────────────────────────────────
    stats = pd.read_csv(_STATS_FILE)
    row = stats[stats["site_uid"] == site_uid]
    if row.empty:
        print(f"[ERROR] {site_uid} not found in site_statistics.csv")
        return None

    start = pd.to_datetime(row.iloc[0]["start_date"]).date()
    end = start + timedelta(days=364)  # exactly 1 year (365 days inclusive)

    # ── resolve basin ─────────────────────────────────────────────────────────
    basin_path = _BASIN_DIR / f"{site_uid}_basin.parquet"
    if not basin_path.exists():
        print(f"[ERROR] No basin file at {basin_path}")
        return None

    basin = gpd.read_parquet(basin_path)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── download + filter ─────────────────────────────────────────────────────
    print(f"\nSite     : {site_uid}")
    print(f"Period   : {start}  →  {end}")
    print(f"Basin    : {basin_path.name}")
    print(f"Basin area: {basin.to_crs('EPSG:26915').geometry.area.iloc[0] / 1e6:.1f} km²")
    print()

    records = []
    for d in tqdm(list(_date_range(start, end)), desc="days", unit="day"):
        zip_path = _download_shapefile(d)
        if zip_path is None:
            continue
        day_gdf = _parse_day_gdf(zip_path, d)
        if day_gdf is None:
            continue
        chunk = _basin_filter(day_gdf, basin)
        if chunk is not None:
            records.append(chunk)

    if not records:
        print("[ERROR] No data returned — check basin coverage and IEM availability")
        return None

    df = pd.concat(records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # ── text summary ──────────────────────────────────────────────────────────
    n_cells = df.groupby(["lon", "lat"]).ngroups
    n_days = df["date"].nunique()
    expected_days = 365

    print(f"\n── DataFrame ──────────────────────────────")
    print(f"Shape           : {df.shape}")
    print(f"Unique grid cells: {n_cells}")
    print(f"Days with data  : {n_days} / {expected_days} expected")
    print(f"Missing days    : {expected_days - n_days}  (IEM had no data)")
    print(f"\nHead:\n{df.head(n_cells + 1).to_string(index=False)}")
    print(f"\nprecip_in_1d summary (all cells, all days):")
    print(df["precip_in_1d"].describe().to_string())

    # ── plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # 1. Daily mean precip across all basin cells ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    daily_mean = df.groupby("date")["precip_in_1d"].mean()
    ax1.bar(daily_mean.index, daily_mean.values, width=1.0, color="#4a90d9", alpha=0.75)
    ax1.set_title("Daily mean precip across basin grid cells")
    ax1.set_ylabel("inches")
    ax1.set_xlabel("Date")

    # 2. Grid cell positions colored by annual mean precip ────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cell_avg = df.groupby(["lon", "lat"])["precip_in_1d"].mean().reset_index()
    sc = ax2.scatter(
        cell_avg["lon"],
        cell_avg["lat"],
        c=cell_avg["precip_in_1d"],
        cmap="YlGnBu",
        s=60,
        edgecolors="#555",
        linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax2, label="Mean daily precip (in)")
    ax2.set_title(f"Basin grid cells  (n={n_cells})")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal")

    # 3. Monthly mean precip (seasonal pattern) ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = df.assign(month=df["date"].dt.to_period("M")).groupby("month")["precip_in_1d"].mean()
    ax3.bar(
        range(len(monthly)),
        monthly.values,
        color="#2e7d32",
        alpha=0.8,
    )
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax3.set_title("Monthly mean daily precip")
    ax3.set_ylabel("inches")

    fig.suptitle(f"IEM rainfall test — {site_uid}  ({start} → {end})", fontsize=13, fontweight="bold")
    plt.show()

    return df


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_UID
    df = run_test(uid)
