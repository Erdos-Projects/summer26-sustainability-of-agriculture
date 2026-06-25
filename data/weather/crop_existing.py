"""Retrofit already-built weather parquets to the current schema, in place.

Brings existing files in line with what the (updated) make_global_weather /
make_basin_weather now produce, without a full rebuild:

  1. Crop to region  — global files (weather_global/) were originally built over
     the entire IEM footprint; drop every row whose (lon, lat) falls outside the
     data.settings.get_region_bbox bbox. (Basin files are already region cells,
     so they are not row-cropped.)

  2. Drop columns + rename — removes lon, lat (constant per global_node_id,
     already in weather_grid), the calendar fields year/month/week/day_of_year
     (derivable from date), and the gridMET variables cut from the feature set
     (spec_humidity, wind_speed, wind_dir, etr, burning_index, energy_release,
     fuel_moisture_100h); and renames srad -> solar_rad, pet -> evapotranspiration.

  3. Shrink — downcast float64 -> float32 and node_id/global_node_id -> int32, and
     re-encode with zstd. This roughly halves the on-disk size at negligible
     precision cost for weather values.

Files are streamed row-group by row-group through a pyarrow ParquetWriter, so peak
memory is one batch, and each output is written to a .tmp file and atomically
renamed. Basin files already split into <uid>_weather_p1.parquet, _p2, ... are
processed too (each part in place); a combined file that is still over the size
limit after shrinking is split. Re-running is a no-op (rows already cropped,
columns already absent, dtypes already narrow).

Usage:
    python data/weather/crop_existing.py                # global + basin, all
    python data/weather/crop_existing.py --year 2012    # one global year (repeatable)
    python data/weather/crop_existing.py --global-only
    python data/weather/crop_existing.py --basin-only
    python data/weather/crop_existing.py --dry-run      # report only, write nothing
"""

import re
import sys
import math
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_THIS_DIR = Path(__file__).resolve().parent
_GLOBAL_DIR = _THIS_DIR / "weather_global"
_DATA_DIR = _THIS_DIR / "weather_data"

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data.settings import get_region_bbox  # noqa: E402

_BATCH_ROWS = 256_000  # rows per streamed batch; ~one IEM grid-month
_COMPRESSION = "zstd"
# Columns removed from existing files: redundant coords/calendar + the gridMET
# variables dropped from the feature set.
_DROP_COLS = [
    "lon", "lat", "year", "month", "week", "day_of_year",
    "spec_humidity", "wind_speed", "wind_dir", "etr",
    "burning_index", "energy_release", "fuel_moisture_100h",
]
# gridMET columns renamed to clearer names.
_RENAME_COLS = {"srad": "solar_rad", "pet": "evapotranspiration"}
_INT32_COLS = {"node_id", "global_node_id"}
_MAX_PART_MB = 90  # keep each shareable file under GitHub's 100 MB limit (headroom)


def _region_mask(table: pa.Table, bbox: tuple) -> pa.Array:
    """Boolean mask: rows whose (lon, lat) lie within the bbox (inclusive)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon, lat = table["lon"], table["lat"]
    return pc.and_(
        pc.and_(pc.greater_equal(lon, min_lon), pc.less_equal(lon, max_lon)),
        pc.and_(pc.greater_equal(lat, min_lat), pc.less_equal(lat, max_lat)),
    )


def _downcast(table: pa.Table) -> pa.Table:
    """float64 -> float32 and node id columns -> int32 (idempotent)."""
    cols, names = [], []
    for n in table.column_names:
        col = table[n]
        if pa.types.is_float64(col.type):
            col = col.cast(pa.float32())
        elif n in _INT32_COLS and pa.types.is_integer(col.type):
            col = col.cast(pa.int32())
        cols.append(col)
        names.append(n)
    return pa.table(cols, names=names)


def _transform(table: pa.Table, bbox: tuple | None) -> pa.Table:
    """Crop to bbox (if given), drop unwanted columns, rename, then downcast."""
    if bbox is not None and "lon" in table.column_names and "lat" in table.column_names:
        table = table.filter(_region_mask(table, bbox))
    drop = [c for c in _DROP_COLS if c in table.column_names]
    if drop:
        table = table.drop(drop)
    table = table.rename_columns([_RENAME_COLS.get(n, n) for n in table.column_names])
    return _downcast(table)


def process_file(path: Path, bbox: tuple | None, dry_run: bool = False) -> tuple[int, int]:
    """Apply _transform to one parquet in place; return (rows_in, rows_out)."""
    pf = pq.ParquetFile(path)
    rows_in = pf.metadata.num_rows

    if dry_run:
        if bbox is None:
            return rows_in, rows_in  # basin files are not row-cropped
        rows_out = 0
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS, columns=["lon", "lat"]):
            rows_out += pc.sum(_region_mask(pa.Table.from_batches([batch]), bbox)).as_py() or 0
        return rows_in, rows_out

    tmp = path.with_suffix(".tmp.parquet")
    writer = None
    rows_out = 0
    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
            table = _transform(pa.Table.from_batches([batch]), bbox)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=_COMPRESSION)
            writer.write_table(table)
            rows_out += table.num_rows
        if writer is not None:
            writer.close()
            writer = None
    except BaseException:  # interrupt/error mid-file -> discard the partial output
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise

    if writer is None and not tmp.exists():
        print(f"  {path.name}: 0 rows after transform — left unchanged")
        return rows_in, 0

    tmp.replace(path)
    return rows_in, rows_out


def split_if_large(path: Path, max_mb: int = _MAX_PART_MB) -> int:
    """Split `path` into row-balanced parts if it exceeds max_mb; else leave it.

    A file over the limit is split into ceil(size / max_mb) parts named
    <stem>_p1.parquet, <stem>_p2.parquet, ... (so {site}_weather.parquet becomes
    {site}_weather_p1.parquet, ...), and the combined file is removed — only the
    parts remain. Parts are contiguous row slices, so concatenating them back in
    order reproduces the original. Returns the number of parts (1 if unchanged).
    """
    size_mb = path.stat().st_size / 1e6
    if size_mb <= max_mb:
        return 1

    n = math.ceil(size_mb / max_mb)
    table = pq.read_table(path)
    rows_per = math.ceil(table.num_rows / n)

    # clear any stale parts from a previous (possibly different-n) split
    for old in path.parent.glob(f"{path.stem}_p*.parquet"):
        old.unlink()

    parts = 0
    for i in range(n):
        chunk = table.slice(i * rows_per, rows_per)
        if chunk.num_rows == 0:
            continue
        pq.write_table(chunk, path.with_name(f"{path.stem}_p{i + 1}.parquet"), compression=_COMPRESSION)
        parts += 1
    path.unlink()  # remove the combined file; reassembled on read from the parts
    return parts


def _is_part(path: Path) -> bool:
    """True for a <uid>_weather_p<N>.parquet split-part file."""
    return re.search(r"_weather_p\d+\.parquet$", path.name) is not None


def _process_dir(paths: list[Path], bbox: tuple | None, label: str, dry_run: bool, split: bool = False) -> None:
    if not paths:
        print(f"{label}: no files found")
        return
    verb = "would keep" if dry_run else "kept"
    how = " (crop + drop columns + shrink)" if bbox else " (drop columns + shrink)"
    if split:
        how += f", split combined files > {_MAX_PART_MB} MB"
    print(f"{label}: {len(paths)} file(s){how}")
    for path in paths:
        rows_in, rows_out = process_file(path, bbox, dry_run=dry_run)
        pct = (rows_out / rows_in) if rows_in else 0
        msg = f"  {path.name}: {rows_in:,} -> {verb} {rows_out:,} rows ({pct:.0%})"
        # only combined files are (re)split; parts are already small and shrinking
        if split and not dry_run and not _is_part(path):
            nparts = split_if_large(path)
            if nparts > 1:
                msg += f"; split into {nparts} parts (> {_MAX_PART_MB} MB)"
        print(msg)


def crop_existing(
    years: list[int] | None = None,
    do_global: bool = True,
    do_basin: bool = True,
    dry_run: bool = False,
) -> None:
    bbox = get_region_bbox()
    print(f"Region bbox (settings): {tuple(round(b, 3) for b in bbox)}")
    print(f"Dropping columns: {_DROP_COLS}")

    if do_global:
        if years:
            paths = [_GLOBAL_DIR / f"global_grid_weather_{y}.parquet" for y in years]
            paths = [p for p in paths if p.exists()]
        else:
            paths = sorted(_GLOBAL_DIR.glob("global_grid_weather_*.parquet"))
        _process_dir(paths, bbox, "Global weather (weather_global)", dry_run)

    if do_basin:
        # both combined files AND already-split parts (the parts must not be skipped)
        paths = sorted(
            set(_DATA_DIR.glob("*_weather.parquet")) | set(_DATA_DIR.glob("*_weather_p*.parquet"))
        )
        _process_dir(paths, None, "Basin weather (weather_data)", dry_run, split=True)


def main(years=None, do_global=True, do_basin=True, dry_run=False):
    crop_existing(years=years, do_global=do_global, do_basin=do_basin, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", action="append", type=int, help="Limit global files to these years (repeatable).")
    parser.add_argument("--global-only", action="store_true", help="Process only the global files.")
    parser.add_argument("--basin-only", action="store_true", help="Process only the basin files.")
    parser.add_argument("--dry-run", action="store_true", help="Report row counts only; write nothing.")
    args = parser.parse_args()
    main(
        years=args.year,
        do_global=not args.basin_only,
        do_basin=not args.global_only,
        dry_run=args.dry_run,
    )
