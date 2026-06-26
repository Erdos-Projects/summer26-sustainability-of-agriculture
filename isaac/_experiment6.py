import sys

sys.path.insert(0, "../")

from data.features import (
    agg_crops,
    agg_surplus,
    agg_weather,
    agg_weather_w_lag,
    daily_nitrate,
    lagged_sensor_nitrate,
    nitrate_rolling,
    nitrate_avg_seasonal,
    nitrate_avg_calendar,
    doy_climatology_pure_signal,
)
from data.transforms import flatten_buckets, merge_on_date, match_seasonal
from data import get_site_ids

from cook import *
from recipes2 import _covariates


def recipe_lagger(lags=[]):
    def recipe(site_uid):
        def lagged(lag):
            wdf = (
                agg_weather(site_uid, edges=[])
                .sort_values("date")
                .set_index("date")
                .asfreq("D")  # regular daily index
                .shift(lag)
            )  # actually lag by `lag` days
            wdf.columns = [f"{c}_lag{lag}" for c in wdf.columns]  # suffix the value columns
            return wdf.reset_index()

        n_daily, parts = _covariates(site_uid)
        parts += [lagged(i) for i in lags]
        out = merge_on_date([n_daily, *parts], spine=n_daily.index)
        return out.dropna(subset=["nitrate_con"]).reset_index(drop=True)

    return recipe


recipe = recipe_lagger([1])

lags = [1, 2, 3, 7, 10, 14, 21, 30]
recipes = {f"Lags {lags[:i]}": recipe_lagger(lags[:i]) for i in range(len(lags))}
print(compare_many(recipes, **FAST_XGB))
