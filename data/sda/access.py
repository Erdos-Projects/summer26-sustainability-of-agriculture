"""Read-only access layer for USDA Soil Data Access (SDA).

Queries the SDA REST API via SQL over HTTP. No local files — all data
is fetched on demand. The main entry point is `query_sda`, which accepts
any valid SDA SQL string. Higher-level helpers translate a lon/lat point
into map unit keys and return pre-built result tables.

SDA SQL reference: https://sdmdataaccess.sc.egov.usda.gov/QueryHelp.aspx
"""

import requests
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt


def query_sda(sql):
    """Submit a SQL query to the SDA REST API and return results as a DataFrame.

    Args:
        sql: Valid SDA SQL string.

    Returns:
        DataFrame with one column per selected field, or an empty DataFrame if
        the query returns no rows.

    Raises:
        requests.HTTPError: On a non-2xx response.
        ValueError: If the response body is empty (usually a malformed query).
    """
    url = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
    payload = {"query": sql, "format": "json+columnname"}

    response = requests.post(url, data=payload)
    response.raise_for_status()

    if not response.text:
        raise ValueError("Empty response from SDA — check your SQL for errors")

    data = response.json()

    table = data.get("Table", [])
    if not table:
        return pd.DataFrame()

    columns = table[0]
    rows = table[1:]
    return pd.DataFrame(rows, columns=columns)


def get_location_data_from_point(lat, lon):
    """Return map unit metadata for the soil map unit at the given WGS-84 point.

    Args:
        lat: Latitude in decimal degrees (WGS-84).
        lon: Longitude in decimal degrees (WGS-84).

    Returns:
        DataFrame with columns: mukey, muname, areasymbol.
    """
    sql = f"""
    SELECT mu.mukey, mu.muname, l.areasymbol
    FROM mapunit mu
    JOIN legend l ON mu.lkey = l.lkey
    WHERE mu.mukey IN (
        SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
            'POINT({lon} {lat})'
        )
    )
    """
    return query_sda(sql)


def get_tables_from_point(lat, lon):
    """Return three soil data tables for the dominant map unit at a WGS-84 point.

    Args:
        lat: Latitude in decimal degrees (WGS-84).
        lon: Longitude in decimal degrees (WGS-84).

    Returns:
        Tuple of (crop_df, horizons_df, restrictions_df).
    """
    mukey = get_location_data_from_point(lat, lon).iloc[0]["mukey"]

    crop_sql = f"""
        SELECT co.compname, co.comppct_r, cc.*
        FROM component co
        JOIN cocropyld cc ON co.cokey = cc.cokey
        WHERE co.mukey = '{mukey}'"""

    horizons_sql = f"""
        SELECT co.compname, co.comppct_r, ch.*
        FROM component co
        JOIN chorizon ch ON co.cokey = ch.cokey
        WHERE co.mukey = '{mukey}'"""

    restrictions_sql = f"""
        SELECT co.compname, co.comppct_r, cr.*
        FROM component co
        JOIN corestrictions cr ON co.cokey = cr.cokey
        WHERE co.mukey = '{mukey}'"""

    return query_sda(crop_sql), query_sda(horizons_sql), query_sda(restrictions_sql)


def get_horizon_table_from_point(lat, lon):
    """Return horizon data for the dominant map unit at a WGS-84 point."""
    mukey = get_location_data_from_point(lat, lon).iloc[0]["mukey"]

    cols = ["cokey", "compname", "comppct_r", "hzdept_r", "hzdepb_r", "claytotal_r",
            "silttotal_r", "sandtotal_r", "om_r", "ksat_r", "awc_r", "ph1to1h2o_r",
            "cec7_r", "dbthirdbar_r", "lep_r", "wsatiated_r", "caco3_r"]
    horizons_sql = f"""
        SELECT co.compname, co.comppct_r, ch.*
        FROM component co
        JOIN chorizon ch ON co.cokey = ch.cokey
        WHERE co.mukey = '{mukey}'"""

    return query_sda(horizons_sql)[cols]


def get_mapunit_polygon(mukey):
    """Return the polygon geometry for a map unit as a GeoJSON FeatureCollection.

    Args:
        mukey: Map unit key, e.g. as returned by get_location_data_from_point.

    Returns:
        GeoJSON FeatureCollection (dict) in WGS-84 coordinates.
    """
    sql = f"""
        SELECT mupolygonkey, musym, mupolygongeo.STAsText() AS wkt
        FROM mupolygon
        WHERE mukey = '{mukey}'
    """
    df = query_sda(sql)
    if df.empty:
        return {"type": "FeatureCollection", "features": []}

    df["geometry"] = df["wkt"].apply(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(df.drop(columns=["wkt"]), geometry="geometry", crs="EPSG:4326")
    return gdf.__geo_interface__


def get_mapunit_geojson_from_point(lat, lon):
    """Return the polygon(s) of the map unit at a WGS-84 point as GeoJSON."""
    mukey = get_location_data_from_point(lat, lon).iloc[0]["mukey"]
    return get_mapunit_polygon(mukey)


def summarize_crop_yields(crop_df):
    """Aggregate crop yield data by crop name.

    Args:
        crop_df: DataFrame as returned by get_tables_from_point.

    Returns:
        DataFrame indexed by cropname with columns: yldunits, nonirryield_r, irryield_r.
    """
    crop_df.nonirryield_r = crop_df.nonirryield_r.apply(lambda x: float(x))
    crop_df = crop_df.drop(columns=["cocropyldkey", "cokey", "vasoiprdgrp", "irryield_h",
                                     "irryield_l", "nonirryield_l", "nonirryield_h",
                                     "comppct_r", "compname", "cropprodindex"])
    return crop_df.groupby("cropname").agg(
        yldunits=("yldunits", "first"),
        nonirryield_r=("nonirryield_r", "sum"),
        irryield_r=("irryield_r", "sum"),
    )
