"""Geometry helpers shared between the map, info, and forecast panels.

The map panel writes the user's current selection ("region of interest") to
the `region-geom` store as a GeoJSON geometry — either a Point (point-select
mode) or a Polygon (rectangle/polygon draw mode). The functions here turn that
selection into whatever shape downstream consumers need.
"""

from shapely.geometry import shape, mapping

# Rough placeholder: ~1km, used to turn a clicked point into a small area for
# forecast purposes. Not geodesically accurate — revisit once the model
# defines what source-area size it expects.
DEFAULT_POINT_BUFFER_DEG = 0.01


def normalize_for_forecast(region_geojson, buffer_deg=DEFAULT_POINT_BUFFER_DEG):
    """Return a Polygon GeoJSON geometry suitable for forecast queries.

    Points (from point-select mode) are buffered into a small disk so the
    model always receives an area. Polygons (from rectangle/polygon draw
    mode) are passed through unchanged.
    """
    geom = shape(region_geojson)
    if geom.geom_type == "Point":
        geom = geom.buffer(buffer_deg)
    return mapping(geom)


def get_downstream_sites(region_geojson, sites_df):
    """Return uids of monitoring sites downstream of the given region.

    Placeholder: identifying downstream sites requires a watershed/flow-
    network dataset that isn't integrated yet. Returns an empty list so
    callers can be written against the final interface now.

    Args:
        region_geojson: Polygon GeoJSON geometry (e.g. from
            normalize_for_forecast) of the source region.
        sites_df: DataFrame of candidate monitoring sites, as returned by
            iwqis_utils.get_iwqis_sites().

    Returns:
        List of site uids downstream of region_geojson.
    """
    return []
