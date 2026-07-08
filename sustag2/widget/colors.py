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
SITE_USGS = {"stroke": "#6b21a8", "fill": "#55a3f7"}
SITE_SELECTED = {"stroke": "darkred", "fill": "red"}

SURPLUS_LOW = (120, 80, 45)  # HSL green (low surplus)
SURPLUS_HIGH = (0, 80, 45)  # HSL red   (high surplus)


# ── Leaflet layer styles (centralised so map_panel stays styling-free) ────────


def basin_style(kind):
    """Leaflet style for a monitoring-site basin layer.

    kind: 'preferred' or 1 -> teal, 2 -> purple, 3 -> violet (matches the basin_type palette).
    """
    palette = {"preferred": (BASIN, 0.15), 1: (BASIN, 0.15), 2: (BASIN_V2, 0.12), 3: (BASIN_V3, 0.12)}
    c, fill_opacity = palette[kind]
    return {
        "pane": "basin-pane",
        "color": c["stroke"],
        "weight": 2,
        "fillOpacity": fill_opacity,
        "fillColor": c["fill"],
        "interactive": False,
    }


def pin_basin_style(method):
    """Leaflet style for a dropped-pin delineated basin. method: 'v1' (NLDI) or 'v3' (D8)."""
    c = {"v1": PIN_BASIN_V1, "v3": PIN_BASIN_V3}[method]
    return {
        "pane": "basin-pane",
        "color": c["stroke"],
        "weight": 2,
        "fillOpacity": 0.15,
        "fillColor": c["fill"],
        "interactive": False,
    }
