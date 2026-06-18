"""
Script that pulls and formats all the USGS water data.
Per-site collector for USGS continuous data.

Builds a 15-minute snapped wide table for one site with one column per available
core pcode (nitrate is always present; others appear only if the site logs them).

Features:
  - tunable chunk period (the OGC continuous endpoint caps each request at 3 years)
  - optional full-lifespan mode: reads the site's period of record from metadata
    and pulls from first observation to now, regardless of how long that is
  - keeps a long-format copy with approval flags for later QA
  - resilient: a failed/empty chunk is logged and skipped, not fatal
"""

from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent  # the directory in which this file is located
SITE_DIR = DATASET_DIR / "sites"  # the directory we save to
LONG_DIR = DATASET_DIR.parent / "IWQIS_archive" / "USGS_QA_DUMP"

SITE_DIR.mkdir(parents=True, exist_ok=True)
LONG_DIR.mkdir(parents=True, exist_ok=True)

UNITS_FILE = DATASET_DIR / "metadata" / "usgs_units.csv"
METADATA_FILE = DATASET_DIR / "metadata" / "usgs_site_metadata.csv"

import logging
import time as _time
import pandas as pd
from dataretrieval import waterdata

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("collector")

GRID = "15min"

CORE_PCODES = {
    "00010": "temp_water",
    "99133": "nitrate_con",
    "00300": "diss_oxy_con",
    "00301": "diss_oxy_sat",
    "00400": "ph",
    "00095": "spec_cond",
    "00060": "discharge",
    "00065": "stage",
}

_CONFIG_FILE = DATASET_DIR / "config" / "pipeline_config.toml"


def _load_usgs_site_list() -> list:
    import tomllib
    with open(_CONFIG_FILE, "rb") as f:
        return tomllib.load(f)["usgs"]["site_list"]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
MAX_ROWS_PER_REQUEST = 50_000
ROWS_PER_DAY_15MIN = 96  # 24h * 4 per param at 15-min cadence
THREE_YEAR_CAP = pd.Timedelta(days=3 * 365)


def auto_chunk_size(n_params, cadence_per_day=ROWS_PER_DAY_15MIN, row_budget=MAX_ROWS_PER_REQUEST, safety=0.85):
    """Largest time window that stays under the row budget for this site.

    n_params: number of parameters the site logs (rows are long-format:
              one row per parameter per timestamp).
    safety:   headroom (0.85 = use 85% of budget) to absorb sub-15-min
              series, overlapping revised series, and cadence jitter.
    Returns a pd.Timedelta, capped at the 3-year API limit.
    """
    n_params = max(1, int(n_params))
    rows_per_day = cadence_per_day * n_params
    days = int((row_budget * safety) // rows_per_day)
    days = max(1, days)
    return min(pd.Timedelta(days=days), THREE_YEAR_CAP)


def make_chunks(start, end, chunk):
    """Yield (chunk_start, chunk_end) windows. `chunk` is a pandas offset alias
    or Timedelta, e.g. '365D', '3Y' (cap), '90D'. Must be <= 3 years for the API."""
    offset = pd.tseries.frequencies.to_offset(chunk) if isinstance(chunk, str) else chunk
    cur = start
    while cur < end:
        nxt = min(cur + offset, end)
        yield cur, nxt
        cur = nxt


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def pull_chunk(site_id, pcodes, c_start, c_end, retries=2, pause=2.0):
    rng = [_fmt(c_start), _fmt(c_end)]
    for attempt in range(retries + 1):
        try:
            df, _ = waterdata.get_continuous(
                monitoring_location_id=site_id,
                parameter_code=list(pcodes),
                time=rng,
            )
            if df is None or len(df) == 0:
                log.info(f"  {rng}: empty")
                return None
            # in pull_chunk, replace the row-count warning with a coverage check
            got_end = pd.to_datetime(df["time"], utc=True).max()
            c_end_utc = c_end if c_end.tzinfo else c_end.tz_localize("UTC")
            if got_end < c_end_utc - pd.Timedelta(days=1):
                log.warning(f"  {rng}: data ends {got_end}, short of {c_end} -- possible truncation")
            return df
        except Exception as e:
            if attempt < retries:
                log.warning(f"  {rng}: {e} -- retry {attempt+1}")
                _time.sleep(pause)
            else:
                log.error(f"  {rng}: FAILED after {retries} retries -- skipping ({e})")
                return None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_site(site_id, start=None, end=None, chunk=None, pcodes=CORE_PCODES, grid=GRID):
    """
    start, end : ISO strings or Timestamps.
    chunk      : window size per request (<= 3 years).
    Returns (wide_df, long_keep_df).
    """

    if chunk is None:
        raise ValueError("Provide a chunk size (use auto_chunk_size to compute one).")

    if start is None or end is None:
        raise ValueError("Provide start and end.")
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    raw_parts = []
    for c_start, c_end in make_chunks(start, end, chunk):
        df = pull_chunk(site_id, pcodes, c_start, c_end)
        if df is not None:
            raw_parts.append(df)

    if not raw_parts:
        log.warning(f"{site_id}: no data in any chunk.")
        return None, None

    raw = pd.concat(raw_parts, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw = raw.drop_duplicates(subset=["parameter_code", "time"])

    # long copy with QA flags, before reshaping
    long_keep = raw[
        ["monitoring_location_id", "parameter_code", "time", "value", "unit_of_measure", "approval_status"]
    ].copy()

    # snap + pivot
    raw["time"] = raw["time"].dt.floor(grid)
    wide = raw.pivot_table(index="time", columns="parameter_code", values="value", aggfunc="first")

    # rename to readable names, order as in pcodes, only those present
    wide = wide.rename(columns=pcodes)
    ordered = [pcodes[pc] for pc in pcodes if pcodes[pc] in wide.columns]
    wide = wide[ordered]
    wide.insert(0, "site_id", site_id)

    return wide, long_keep


def generate_metadata(generate_relevant_sites=False):
    if generate_relevant_sites:
        nitrogen_pcodes = ["99133", "99137", "83554", "00631"]
        state = "Iowa"

        # Join the list into a comma-separated string
        pcodes_str = ",".join(nitrogen_pcodes)

        # Query the time-series catalog
        timeseries_df, _ = waterdata.get_time_series_metadata(
            state_name=state, parameter_code=pcodes_str, skip_geometry=False
        )

        # Extract the unique IDs
        monitoring_location_ids = timeseries_df["monitoring_location_id"].unique().tolist()
    else:
        monitoring_location_ids = _load_usgs_site_list()

    all_meta, _ = waterdata.get_time_series_metadata(
        monitoring_location_id=monitoring_location_ids,  # the 26 identified
        skip_geometry=False,
    )
    # no parameter_code filter -> returns every series at each site

    return all_meta


def qa_summary(long_keep):
    qa = (
        long_keep.groupby(["parameter_code", "approval_status"])
        .agg(
            begin=("time", "min"),
            end=("time", "max"),
            n=("time", "size"),
        )
        .reset_index()
        .sort_values(["parameter_code", "begin"])
    )
    return qa


def update_measures(long_keep, pcodes=CORE_PCODES, path=UNITS_FILE):
    """Maintain a parameter_code -> unit lookup across all sites.

    Accumulates into one shared CSV. Warns if a pcode appears with a unit
    different from what's already recorded (e.g. a stage series in ft vs m).
    """
    # distinct (pcode, unit) seen in this site's data
    seen = long_keep[["parameter_code", "unit_of_measure"]].dropna().drop_duplicates().copy()
    seen["name"] = seen["parameter_code"].map(pcodes)

    # load existing lookup if present
    if path.exists():
        existing = pd.read_csv(path, dtype={"parameter_code": str})
    else:
        existing = pd.DataFrame(columns=["parameter_code", "name", "unit_of_measure"])

    seen["parameter_code"] = seen["parameter_code"].astype(str)
    seen = seen[["parameter_code", "name", "unit_of_measure"]]

    combined = pd.concat([existing, seen], ignore_index=True).drop_duplicates(
        subset=["parameter_code", "unit_of_measure"]
    )

    # flag any pcode that now has more than one unit
    dupes = combined[combined.duplicated("parameter_code", keep=False)]
    if not dupes.empty:
        for pc, g in dupes.groupby("parameter_code"):
            units = ", ".join(g["unit_of_measure"].astype(str))
            log.warning(f"unit conflict for {pc} ({pcodes.get(pc,'?')}): {units}")

    combined = combined.sort_values("parameter_code")
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Per-site helpers — module-level so they are importable and testable
# ---------------------------------------------------------------------------
def filename(site_id: str) -> Path:
    return SITE_DIR / f"{site_id}_all_data.parquet"


def get_lifespan(site_id: str, meta: "pd.DataFrame"):
    mask = (meta["monitoring_location_id"] == site_id) & (meta["parameter_code"].astype(str) == "99133")
    start = pd.to_datetime(meta[mask]["begin"], utc=True, errors="coerce").min()
    end   = pd.to_datetime(meta[mask]["end"],   utc=True, errors="coerce").max()
    return start, end


def get_update_ranges(site_id: str, meta: "pd.DataFrame") -> list:
    """Return [(start, end)] gaps that need fetching; [] if up to date."""
    _TOL = pd.Timedelta(days=2)
    path = filename(site_id)
    meta_start, meta_end = get_lifespan(site_id, meta)

    if pd.isna(meta_start) or pd.isna(meta_end):
        log.warning(f"{site_id}: no nitrate lifespan in metadata, skipping")
        return []

    if not path.exists():
        return [(meta_start, meta_end)]

    try:
        idx = pd.read_parquet(path, columns=[]).index
    except Exception:
        print(f"File {path} unreadable, will re-fetch full range.")
        return [(meta_start, meta_end)]

    if idx.empty:
        return [(meta_start, meta_end)]

    def _utc(ts):
        ts = pd.Timestamp(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    file_start = _utc(idx.min())
    file_end   = _utc(idx.max())

    ranges = []
    gap_before = file_start - meta_start
    gap_after  = meta_end  - file_end
    if gap_before > _TOL:
        print(f"{site_id}: {gap_before.days}d missing at start → fetching {meta_start:%Y-%m-%d} to {file_start:%Y-%m-%d}")
        ranges.append((meta_start, file_start))
    if gap_after > _TOL:
        print(f"{site_id}: {gap_after.days}d missing at end → fetching {file_end:%Y-%m-%d} to {meta_end:%Y-%m-%d}")
        ranges.append((file_end, meta_end))

    return ranges


def params_from_meta(site_id: str, meta: "pd.DataFrame") -> int:
    sub = meta[(meta["monitoring_location_id"] == site_id) & meta["parameter_code"].astype(str).isin(CORE_PCODES)]
    return max(1, sub["parameter_code"].nunique())


# ---------------------------------------------------------------------------
def main(api_keys, extra_filter=None):
    """Builds the USGS data site by site. Creates the files
    - `usgs-site-metadata.csv`: metadata for all relevant sites
    - `<site-id>_all_data.csv`: full timeseries data for every relevant site

    Parameters
    ----------
    api_key : dict{str : str}
        Dictionary containing the api keys for data access.
        Must contain one for the key "usgs"

    Returns
    -------
    _type_
        _description_
    """
    import os

    if extra_filter is None:
        extra_filter = []

    os.environ["API_USGS_PAT"] = api_keys["usgs"]

    # metadata -------------------------------------------------------------
    site_list = _load_usgs_site_list()
    try:
        meta = pd.read_csv(METADATA_FILE, dtype={"parameter_code": str})
        meta_sites = set(meta["monitoring_location_id"].unique())
        if meta_sites != set(site_list):
            meta = generate_metadata()
    except FileNotFoundError:
        meta = generate_metadata()

    print(f"\n{meta.monitoring_location_id.nunique()} sites in metadata\n")
    site_ids = (
        meta[
            meta["monitoring_location_id"].isin(site_list) &
            ~meta["monitoring_location_id"].isin(extra_filter)
        ]["monitoring_location_id"].unique().tolist()
    )

    completed_site_ids = []
    for site_id in site_ids:
        path = filename(site_id)
        update_ranges = get_update_ranges(site_id, meta)

        if not update_ranges:
            print(f"Site {site_id} up to date, skipping")
            completed_site_ids.append(site_id)
            continue

        n_params = params_from_meta(site_id, meta)
        chunk = auto_chunk_size(n_params)

        # Load existing data so we can merge incremental fetches into it
        existing_wide = None
        if path.exists():
            try:
                existing_wide = pd.read_parquet(path)
            except Exception:
                log.warning(f"{site_id}: could not read existing file, re-fetching full range")
                update_ranges = [get_lifespan(site_id, meta)]

        wide_parts = [existing_wide] if existing_wide is not None else []
        long_parts = []

        for r_start, r_end in update_ranges:
            print(f"  {site_id}: fetching {r_start:%Y-%m-%d} to {r_end:%Y-%m-%d}  ({n_params} params, chunk {chunk.days}d)")
            wide_chunk, long_chunk = build_site(site_id, start=r_start, end=r_end, chunk=chunk)
            if wide_chunk is not None:
                wide_parts.append(wide_chunk.rename(columns={"site_id": "site_uid"}))
            if long_chunk is not None:
                long_parts.append(long_chunk)

        if not wide_parts:
            print(f"  {site_id}: no data returned, skipping")
            continue

        wide = pd.concat(wide_parts)
        wide = wide[~wide.index.duplicated(keep="last")].sort_index()
        wide.index.name = "datetime"

        # display the results -----------------------------------------
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)

        print(f"\n=== {site_id}  shape={wide.shape}  grid={GRID} ===")
        print(f"span: {wide.index.min()} -> {wide.index.max()}\n")
        print(wide.head().to_string())

        print("\n--- column fill ---")
        counts = wide.drop(columns="site_uid").notna().sum()
        print(
            pd.DataFrame(
                {
                    "non_null": counts,
                    "pct": (counts / len(wide) * 100).round(1),
                }
            ).to_string()
        )

        empty = wide.drop(columns="site_uid").isna().all(axis=1).sum()
        print(f"\nfully-empty rows: {empty} / {len(wide)}")

        out = filename(site_id)
        wide.to_parquet(out)
        print(f"\nWrote {out}")

        if long_parts:
            long_keep = pd.concat(long_parts, ignore_index=True)
            qa = qa_summary(long_keep)
            qa.to_csv(LONG_DIR / f"{site_id}_qa.csv", index=False)
            update_measures(long_keep)

        completed_site_ids.append(site_id)

    # write the metadata file to a csv
    meta = meta[meta.monitoring_location_id.isin(completed_site_ids) == True]
    meta.to_csv(METADATA_FILE)
    print(f"Saved full metadata from {len(completed_site_ids)} sites to {METADATA_FILE}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tomllib

    ROOT = Path(__file__).resolve().parents[1]
    API_KEYS = ROOT / "api-keys.toml"

    with open(API_KEYS, "rb") as f:
        keys = tomllib.load(f)

    main(keys)
