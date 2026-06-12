"""Seam between the Dash front end and the nitrate exceedance forecast model.

This module defines the contract the front end relies on. The model itself
is not implemented yet — `forecast_exceedance` raises NotImplementedError
until it's wired up. Everything in components/forecast_panel.py is written
against this interface so the model can be dropped in later without touching
the Dash code.
"""

from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Per-site forecast outcome."""

    exceeds_threshold: bool
    probability: float
    eta_days: int | None  # estimated days until the 10 mg/L threshold is reached


def forecast_exceedance(region_geojson, nitrogen_surplus, target_uids):
    """Forecast whether nitrate concentration at each target site exceeds
    10 mg/L within one month, given an average nitrogen surplus over the
    source region.

    Args:
        region_geojson: Polygon GeoJSON geometry of the source region (see
            data.geo_utils.normalize_for_forecast).
        nitrogen_surplus: Average nitrogen surplus over the region, kg/ha.
        target_uids: IWQIS site uids to forecast for — typically the result
            of data.geo_utils.get_downstream_sites.

    Returns:
        dict mapping site uid -> ForecastResult.
    """
    raise NotImplementedError("Nitrate forecast model is not yet implemented")
