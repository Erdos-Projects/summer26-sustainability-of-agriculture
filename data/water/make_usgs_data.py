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
LONG_DIR = DATASET_DIR / "LONG_data_dump"

SITE_DIR.mkdir(parents=True, exist_ok=True)
LONG_DIR.mkdir(parents=True, exist_ok=True)

UNITS_FILE = DATASET_DIR / "usgs_units.csv"
METADATA_FILE = DATASET_DIR / "usgs_site_metadata.csv"

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

SITE_LIST = [
    "USGS-05418400",
    "USGS-05474500",
    "USGS-06604440",
    "USGS-05455100",
    "USGS-05418110",
    "USGS-415959091441301",
    "USGS-05484000",
    "USGS-05480986",
    "USGS-05412500",
    "USGS-05480603",
    "USGS-06817000",
    "USGS-05420500",
    "USGS-05418720",
    "USGS-05464500",
    "USGS-05464420",
    "USGS-06808500",
    "USGS-05451210",
    "USGS-06603750",
    "USGS-05484500",
    "USGS-05482500",
    "USGS-05482300",
    "USGS-05481000",
    "USGS-05465500",
    "USGS-05482000",
    "USGS-05483600",
    "USGS-05464475",
]

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
        monitoring_location_ids = SITE_LIST

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


def main(api_keys):
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

    os.environ["API_USGS_PAT"] = api_keys["usgs"]

    # metadata -------------------------------------------------------------
    try:
        # making parameter_code a string to avoid it being interrpreted as an integer
        meta = pd.read_csv(METADATA_FILE, dtype={"parameter_code": str})
        if meta["monitoring_location_id"].nunique() < 27:
            meta = generate_metadata(generate_relevant_sites=True)
    except FileNotFoundError:
        meta = generate_metadata()

    site_ids = meta["monitoring_location_id"].unique().tolist()

    # helper function for defining filename --------------------------------
    def filename(site_id):
        return SITE_DIR / f"{site_id}_all_data.parquet"

    # get start and end dates ----------------------------------------------
    def get_lifespan(site_id):
        filter = (meta["monitoring_location_id"] == site_id) & (meta["parameter_code"].astype(str) == "99133")
        start = pd.to_datetime(meta[filter]["begin"], utc=True, errors="coerce").min()
        end = pd.to_datetime(meta[filter]["end"], utc=True, errors="coerce").max()
        return start, end

    # check whether dataset for site already exists ------------------------
    def site_exists(site_id):
        path = filename(site_id)
        if not path.exists():
            print(f"File {path} not found. Retrieving data.")
            return False
        try:
            # read only the index — avoids loading all columns
            idx = pd.read_parquet(path, columns=[]).index
        except Exception:
            print(f"File {path} unreadable. Retrieving data.")
            return False
        if idx.empty:
            print(f"File {path} has no data. Retrieving data.")
            return False
        s1, e1 = get_lifespan(site_id)
        s2, e2 = idx.min(), idx.max()

        # tolerate snap/boundary offset; "done" if file spans ~the metadata window
        lifespan_match = abs((s2 - s1).total_seconds()) < 86400 and abs((e2 - e1).total_seconds()) < 86400
        if not lifespan_match:
            print(f"Lifespan mismatch. Retrieving data.")
        return lifespan_match

    def params_from_meta(site_id):
        sub = meta[(meta["monitoring_location_id"] == site_id) & meta["parameter_code"].astype(str).isin(CORE_PCODES)]
        return max(1, sub["parameter_code"].nunique())

    completed_site_ids = []
    for site_id in site_ids:
        if site_exists(site_id=site_id):
            print(f"Site {site_id} already exists at {filename(site_id)}, skipping")
        else:
            start, end = get_lifespan(site_id)
            n_params = params_from_meta(site_id)
            chunk = auto_chunk_size(n_params)

            print(f"Site {site_id} doesn't exist, creating it...")
            print(f"   start = {start}")
            print(f"     end = {end}")
            print(f"  params = {n_params} -> chunk {chunk.days}d")
            # build the data ---------------------------------------------
            wide, long_keep = build_site(
                site_id,
                start=start,
                end=end,
                chunk=chunk,
            )

            if wide is None:
                print(f"  {site_id}: no data returned, skipping")
                continue

            # display the results -----------------------------------------
            pd.set_option("display.width", 200)
            pd.set_option("display.max_columns", 20)

            print(f"\n=== {site_id}  shape={wide.shape}  grid={GRID} ===")
            print(f"span: {wide.index.min()} -> {wide.index.max()}\n")
            print(wide.head().to_string())

            print("\n--- column fill ---")
            counts = wide.drop(columns="site_id").notna().sum()
            print(
                pd.DataFrame(
                    {
                        "non_null": counts,
                        "pct": (counts / len(wide) * 100).round(1),
                    }
                ).to_string()
            )

            empty = wide.drop(columns="site_id").isna().all(axis=1).sum()
            print(f"\nfully-empty rows: {empty} / {len(wide)}")

            # write the main file — align schema with IWQIS (site_uid, datetime index)
            out = filename(site_id)
            wide = wide.rename(columns={"site_id": "site_uid"})
            wide.index.name = "datetime"
            wide.to_parquet(out)

            # update the units and create the qa summary -----------------------------
            qa = qa_summary(long_keep)
            qa.to_csv(LONG_DIR / f"{site_id}_qa.csv", index=False)
            update_measures(long_keep)
            print(f"\nWrote {out}")

        # add siteid in either event
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
