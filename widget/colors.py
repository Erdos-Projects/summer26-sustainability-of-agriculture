"""Map layer color palette.

Each entry is a dict with 'stroke' and/or 'fill' keys (CSS color strings).
Surplus gradient is stored as (hue, saturation, lightness) HSL tuples so the
interpolation in map_panel stays in the same color space.
"""

HYDRO = {"stroke": "#2563eb", "fill": "#3b82f6"}

BASIN = {"stroke": "#0d9488", "fill": "#0d9488"}
BASIN_V2 = {"stroke": "purple", "fill": "purple"}
BASIN_V3 = {"stroke": "#7c3aed", "fill": "#7c3aed"}

PIN_BASIN_V1 = {"stroke": "#f97316", "fill": "#f97316"}
PIN_BASIN_V3 = {"stroke": "#0891b2", "fill": "#0891b2"}

RAIN_GRID = {"stroke": "#0284c7", "fill": "#38bdf8"}
IEM_BBOX = {"stroke": "#f97316"}

SITE_DEFAULT = {"stroke": "darkgreen", "fill": "limegreen"}
SITE_USGS = {"stroke": "#6b21a8", "fill": "#a855f7"}
SITE_SELECTED = {"stroke": "darkred", "fill": "red"}

SURPLUS_LOW = (120, 80, 45)  # HSL green (low surplus)
SURPLUS_HIGH = (0, 80, 45)  # HSL red   (high surplus)
