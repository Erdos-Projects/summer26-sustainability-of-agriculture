import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask
from make_surplus_2 import _RAW_DIR, _MERGED_FILE, EQUAL_AREA_CRS, _get_grid_index
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
from data import basins

site_uid = "WQS0115"
merged = pd.read_parquet(_MERGED_FILE)
basin = basins.get_basin(site_uid)
poly = basin.to_crs(EQUAL_AREA_CRS).geometry.union_all()

# --- OLD METHOD (point-in-polygon) ---
minx, miny, maxx, maxy = poly.bounds
candidates = merged[(merged["x"] >= minx) & (merged["x"] <= maxx) & (merged["y"] >= miny) & (merged["y"] <= maxy)]
unique_pixels = candidates.drop_duplicates("pixel_id")
pts = gpd.GeoSeries(
    gpd.points_from_xy(unique_pixels["x"], unique_pixels["y"]),
    index=unique_pixels.index, crs=EQUAL_AREA_CRS,
)
old_ids = set(unique_pixels.loc[pts.within(poly), "pixel_id"])

# --- NEW METHOD (rasterize) ---
transform, height, width = _get_grid_index()
mask = geometry_mask([poly], out_shape=(height, width), transform=transform, invert=True)
new_ids = set(np.flatnonzero(mask.ravel()))

# --- COMPARE ---
print(f"old: {len(old_ids)} pixels")
print(f"new: {len(new_ids)} pixels")
print(f"in old, missing from new: {len(old_ids - new_ids)}")
print(f"in new, not in old:       {len(new_ids - old_ids)}")
print(f"exact match: {old_ids == new_ids}")