"""Read-only access layer for rain data in data/rain/."""

from typing import Callable
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_RAIN_DIR = _THIS_DIR / "rain_data"
_GRID_DIR = _THIS_DIR / "rain_grid"
_GLOBAL_GRID_FILE = _GRID_DIR / "global_rain_grid.parquet"
_BASIN_DIR = _THIS_DIR.parent / "basins" / "basin_data"


def get_rain_grid(site_uid: str) -> gpd.GeoDataFrame:
    """Load the rain grid (Voronoi target cells) for a site.

    Returns a GeoDataFrame with columns node_id, global_node_id, x, y (EPSG:5070),
    lat, lon, cell_area, dist_to_sensor, frac_cell_in_basin, geometry. This is the
    shared spatial grid the surplus and crop aggregates are built on; node_id is
    the basin-local join key, global_node_id is the canonical IEM cell index
    (shared across basins, so overlapping basins match on it). dist_to_sensor is
    the metres-of-flow from the node centre to the monitoring sensor;
    frac_cell_in_basin is the fraction of the cell's area inside the basin, in
    [0, 1].

    Raises FileNotFoundError if it has not been generated yet (run make_rain.py).
    """
    path = _GRID_DIR / f"{site_uid}_rain_grid.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No rain grid for {site_uid}. Run make_rain.py to generate {path.name}.")
    return gpd.read_parquet(path)


def get_global_rain_grid() -> pd.DataFrame:
    """Load the global rain grid (cell -> sites containing it).

    Returns a DataFrame with columns global_node_id, contained_in_sites
    (list of site_uids whose rain grid includes the cell), n_sites, lat, lon
    (cell centroid, WGS84). Only cells contained in at least one preferred basin
    appear; global_node_id is the canonical IEM cell index shared across basins.

    Raises FileNotFoundError if it has not been generated yet (run make_rain.py).
    """
    if not _GLOBAL_GRID_FILE.exists():
        raise FileNotFoundError(
            f"No global rain grid. Run make_rain.py (build_grids) to generate {_GLOBAL_GRID_FILE.name}."
        )
    return pd.read_parquet(_GLOBAL_GRID_FILE)


def get_rain(site_uid: str) -> pd.DataFrame:
    """Load the rainfall parquet for a site.

    Parameters
    ----------
    site_uid : str
        The unique site identifier.

    Returns
    -------
    DataFrame with columns: date, node_id, global_node_id, lon, lat,
    precip_in_1d, year, month, day_of_year, week.
    node_id joins each row to its cell in the rain grid (get_rain_grid);
    global_node_id is the canonical IEM cell index, shared across basins.

    Raises FileNotFoundError if the parquet has not been generated yet
    (run make_rain.py to produce it).
    """
    path = _RAIN_DIR / f"{site_uid}_rain.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No rain data for {site_uid}. " f"Run make_rain.py to generate {path.name}.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_by_interval(
    site_uid: str | None = None,
    df: pd.DataFrame | None = None,
    value_col: str = "precip_in_1d",
    interval: str = "3D",
    agg_func: str | Callable = "sum",
) -> pd.DataFrame:
    """Aggregate a site's rainfall data by a temporal interval, per grid cell.

    Spatial structure is preserved: the output has one row per (grid cell,
    period), where each period covers `interval` days.  The date column marks
    the start of each period.  This allows downstream models to use lon/lat
    as spatial features alongside the aggregated precipitation value.

    For example, with interval='3D' and a basin covering 40 grid cells, each
    3-day period produces 40 rows — one per cell.

    Parameters
    ----------
    site_uid : str, optional
        Site identifier.  Used to load data when df is not provided.
    df : DataFrame, optional
        Pre-loaded rain DataFrame (output of get_rain).  Takes
        precedence over site_uid if both are supplied.
    value_col : str
        Column to aggregate.  Defaults to 'precip_in_1d'.
    interval : str
        Pandas offset alias, e.g. '1D', '3D', '1W', '1MS', '3MS', '1YS'.
    agg_func : str or callable
        Aggregation function.  Defaults to 'sum' — the natural choice for
        precipitation accumulation over a period.

    Returns
    -------
    DataFrame with columns: date, lon, lat, precip_<interval>.
    Dates are spaced by `interval`; each row is one grid cell for one period.

    Example
    -------
    df_3d  = aggregate_by_interval("USGS-05482500", interval="3D")
    df_1w  = aggregate_by_interval(df=rain_df, interval="1W")
    """
    if df is None:
        if site_uid is None:
            raise ValueError("provide either df or site_uid")
        df = get_rain(site_uid)

    result = (
        df.set_index("date")
        .groupby(["lon", "lat"])[value_col]
        .resample(interval)
        .agg(agg_func)
        .reset_index()
        .rename(columns={value_col: f"precip_{interval.lower()}"})
    )

    return result


def plot_rain(site_uid: str, show: bool = True) -> plt.Figure:
    """Three-panel rainfall summary figure for a single monitoring site.

    Panels
    ------
    Top    : bar chart of daily mean precip across all basin grid cells
    Bottom left  : scatter of grid cell positions, coloured by annual mean precip
    Bottom right : bar chart of monthly mean daily precip

    Parameters
    ----------
    site_uid : str
        Site identifier matching a file in data/rain/rain/.
    show : bool
        Call plt.show() before returning.  Set False when embedding in a
        notebook or other display context that handles rendering itself.

    Returns
    -------
    matplotlib Figure
    """
    df = get_rain(site_uid)

    n_cells = df.groupby(["lon", "lat"]).ngroups
    start = df["date"].min().date()
    end = df["date"].max().date()

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ── daily mean precip (full-width top panel) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    daily_mean = df.groupby("date")["precip_in_1d"].mean()
    ax1.bar(daily_mean.index, daily_mean.values, width=1.0, color="#4a90d9", alpha=0.75)
    ax1.set_title("Daily mean precip across basin grid cells")
    ax1.set_ylabel("inches")
    ax1.set_xlabel("Date")

    # ── grid cell positions coloured by annual mean precip ────────────────────
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
        zorder=3,
    )
    plt.colorbar(sc, ax=ax2, label="Mean daily precip (in)")

    basin_path = _BASIN_DIR / f"{site_uid}_basin1.parquet"
    if basin_path.exists():
        gpd.read_parquet(basin_path).plot(
            ax=ax2,
            facecolor="none",
            edgecolor="#e53935",
            linewidth=1.2,
            zorder=2,
        )

    ax2.set_title(f"Basin grid cells  (n={n_cells})")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect("equal")

    # ── monthly mean precip ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    monthly = df.assign(month_period=df["date"].dt.to_period("M")).groupby("month_period")["precip_in_1d"].mean()
    ax3.bar(range(len(monthly)), monthly.values, color="#2e7d32", alpha=0.8)
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax3.set_title("Monthly mean daily precip")
    ax3.set_ylabel("inches")

    fig.suptitle(
        f"IEM rainfall — {site_uid}  ({start} → {end})",
        fontsize=13,
        fontweight="bold",
    )

    if show:
        plt.show()

    return fig
