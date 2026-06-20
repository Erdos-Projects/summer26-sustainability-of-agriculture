"""Clip manually-downloaded national CDL rasters to a cached regional subset.

This is a standalone preprocessing tool. It is NOT imported by make_crops.py —
make_crops simply assumes the clips it produces already exist in
crops_raw/clipped/. Run this first (whenever you add national files) to build or
refresh that cache.

Source (manual download — not scripted)
    Download the national CONUS Cropland Data Layer GeoTIFFs from the NASS
    National Download page and unzip them into crops_raw/national/:

        https://www.nass.usda.gov/Research_and_Science/Cropland/Release/
            datasets/{YEAR}_30m_cdls.zip      (2008-2025, 30 m)
            datasets/{YEAR}_10m_cdls.zip      (2024+, 10 m)

    Each unzips to a file named like  {YEAR}_30m_cdls.tif  (CONUS, EPSG:5070).

What this does
    Clip every national raster down to a generous bounding box defined in WGS84
    (BBOX_WGS84 below) and cache the result in crops_raw/clipped/. The box is
    deliberately larger than the current basins: basins may be re-assigned
    slightly, and we don't want a basin edit to force re-clipping the multi-GB
    national files. We keep all of Iowa + a margin, throwing away the rest of
    CONUS.

Output (crops_raw/clipped/  — gitignored)
    cdl_clip_{year}.tif    class codes, EPSG:5070, windowed to BBOX_WGS84

Usage
-----
    python clip_crops.py                 # clip every national file found
    python clip_crops.py --year 2020     # just one year (repeatable)
    python clip_crops.py --force         # re-clip even if cached
"""

import requests
import xml.etree.ElementTree as ET
import argparse
import os
import re
import sys
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

_THIS_DIR = Path(__file__).resolve().parent
_RAW_DIR = _THIS_DIR / "crops_raw"
_NATIONAL_DIR = _RAW_DIR / "national"  # put manually-downloaded national .tif files here
_DOWNLOAD_DIR = _RAW_DIR / "downloaded"  # raw CropScape downloads, before clipping
_CLIP_DIR = _RAW_DIR / "clipped"  # cached regional clips land here

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data.settings import get_region_bbox, get_config

# ── Region to keep, in WGS84 (lon/lat) ───────────────────────────────────────
# Shared region from pipeline_config.toml [region].bbox_wgs84 as
# (min_lon, min_lat, max_lon, max_lat). Edit that file to adjust the clip margin.
BBOX_WGS84 = get_region_bbox()

# Filenames encode year and resolution: {year}_{res}m_cdls.
_FNAME_RE = re.compile(r"(\d{4})_(\d+)m_cdls", re.IGNORECASE)


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _clip_path(year: int) -> Path:
    return _CLIP_DIR / f"cdl_clip_{year}.tif"


def _download_path(year: int) -> Path:
    return _DOWNLOAD_DIR / f"cdl_download_{year}.tif"


def find_national_files() -> dict[int, Path]:
    """Map year -> national CDL GeoTIFF found anywhere under crops_raw/national/.

    Accepts both 30 m and 10 m files; if both exist for a year, prefers 30 m
    (keeps the time series at one resolution). Searches recursively because the
    NASS zips unzip into a per-year subfolder.
    """
    found: dict[int, Path] = {}
    for path in _NATIONAL_DIR.rglob("*_cdls.tif"):
        m = _FNAME_RE.search(path.name)
        if not m:
            continue
        year, res = int(m.group(1)), int(m.group(2))
        # prefer 30 m when a year has both
        if year in found and res != 30:
            continue
        found[year] = path
    return found


def clip_to_bbox(src_path: Path, out_path: Path) -> tuple[int, int]:
    """Window a national CDL raster down to BBOX_WGS84 and write it out.

    Returns (width, height) of the clip. Preserves CRS, class codes, nodata and
    the CDL colormap; compresses output with LZW.
    """
    with rasterio.open(src_path) as src:
        # Reproject the WGS84 box into the raster's CRS (EPSG:5070). densify_pts
        # follows the curved Albers edges so the envelope safely encloses the box.
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *BBOX_WGS84, densify_pts=21)
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()

        # The source must fully contain the clip region. A smaller source (e.g. a
        # partial download) would otherwise be silently clamped to a too-small
        # clip, breaking the uniform extent every clip is meant to share.
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > src.width
            or window.row_off + window.height > src.height
        ):
            raise ValueError(
                f"Clip region is not fully contained in source '{src_path.name}'. "
                f"The source raster is smaller than the region {BBOX_WGS84} — "
                f"re-download the full region before clipping."
            )

        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=transform,
            compress="lzw",
        )
        try:
            colormap = src.colormap(1)
        except ValueError:
            colormap = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        if colormap:
            dst.write_colormap(1, colormap)
    return int(window.width), int(window.height)


def download_source(year):
    raw_path = _download_path(year)

    # Skip the (slow, flaky) download when the raw file is already on disk — just
    # re-clip it. Delete the raw file to force a fresh download.
    if not raw_path.exists():
        minx, miny, maxx, maxy = get_region_bbox(albers=True)
        albers_bbox = f"{minx:.0f},{miny:.0f},{maxx:.0f},{maxy:.0f}"
        url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile" f"?year={year}&bbox={albers_bbox}"

        # Step 1: request XML response containing GeoTIFF download URL
        resp = requests.get(url, timeout=60)

        if resp.status_code == 500:
            print("500 Server Error — common causes:")
            print("  - bbox outside continental US")
            print("  - CDL not available for this year/region")
            print(f"Raw response: {resp.text[:300]}")
            return None

        root = ET.fromstring(resp.content)

        # The download URL is in a <returnURL> element. {*} matches the element in
        # any namespace or none, so no fallback query is needed (and `or` on an
        # Element is deprecated + buggy for childless elements).
        url_elem = root.find(".//{*}returnURL")
        if url_elem is None or not url_elem.text:
            print("ERROR: Could not parse GeoTIFF URL from CDL response.")
            print(f"Raw response: {resp.text[:400]}")
            return None

        tiff_url = url_elem.text.strip()
        print(f"GeoTIFF URL: {tiff_url}\n")

        # Step 2: stream the raw download to the downloads dir (no full copy in
        # memory). Write to a temp name and rename on success, so an interrupted
        # download never leaves a truncated file that looks complete.
        tmp_path = raw_path.with_suffix(".tif.part")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(tiff_url, timeout=60, stream=True) as tiff_resp:
            tiff_resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in tiff_resp.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                    f.write(chunk)
        os.replace(tmp_path, raw_path)  # atomic; the final file only appears once complete
    else:
        print(f"  {year}: using existing download {raw_path.name}")

    return raw_path


def main(force: bool = False, years: list = None, download: bool = False) -> None:
    if years is None:
        y1 = get_config()["crops"]["year_start"]
        y2 = get_config()["crops"]["year_end"]
        years = set(range(y1, y2 + 1))

    done = [y for y in years if _clip_path(y).exists()]

    if done != [] and force == False:
        print(
            f"Years\n  {done}\nalready exist in {_clip_path(done[0]).parts[-3:]}. Rerun with the flag --force to rebuild them.\n"
        )

    years = set(years).difference(set(done))

    if BBOX_WGS84 is None:
        raise ValueError("Set BBOX_WGS84 (min_lon, min_lat, max_lon, max_lat) before running.")

    _CLIP_DIR.mkdir(parents=True, exist_ok=True)

    files = find_national_files()
    missing = years.difference(set(files))

    if not missing and download == False:
        print(f"Missing national CDL .tif files for years {missing}.")
        if download == False:
            print("Download + unzip the national rasters to {_NATIONAL_DIR} first, e.g.:")
            print(
                "  <YEAR>_30m_cdls.tif    from\n  https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php"
            )
            print("or rerun this script with the --download flag to download directly.")
            return

    def clip_and_save(year, file_to_clip):
        out = _clip_path(year)
        if out.exists() and not force:
            print(f"  {year}: exists already ({_human(out.stat().st_size)})")
            return

        w, h = clip_to_bbox(file_to_clip, out)
        print(f"  {year}: {file_to_clip.name} -> {out.name}  ({w}x{h} px, {_human(out.stat().st_size)})")

    print(f"Region (WGS84): {BBOX_WGS84}")
    print(f"Clipping {len(years)} file(s) to this region.")
    for year in years:
        # if source a national file to be clipped
        if year in files:
            clip_and_save(year, files[year])
        elif year not in files and download:
            clip_and_save(year, download_source(year))
        else:
            print(f"  {year}: skipped.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-clip even if the cached clip exists.")
    parser.add_argument("--download", action="store_true", help="Download years which fail")
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        metavar="YYYY",
        help="Process only this year (repeatable, e.g. --year 2020 --year 2021).",
    )
    args = parser.parse_args()
    main(force=args.force, years=args.year, download=args.download)
