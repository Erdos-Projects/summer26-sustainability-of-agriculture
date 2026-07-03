No AI was used in the preparation of this document.

## Target

The two main targets are derived from a collection of 162 water-borne nitrate sensors spread out across Iowa's hydrological network recording nitrate (NO2 + NO3) concentration measurements (mg/L) in 5-15 minute intervals. Each of these sensors began reporting at a different date between 2008 and 2025, and many sensors were decommissioned. There are also significant data gaps in most of the nitrate timeseries in which a sensor was turned off for a time (or malfunctioned and was replaced) before being turned back on again. This means the quality of these timeseries vary significantly from sensor to sensor -- some are mostly nan, some span time periods of 1 year. To standardize clean the data we filtered out sites that had a NaN reporting rate above 50% or a total lifespan of under 3.92 years (3.92 was chosen beause there was one sensor with high quality data we liked a lot). We then manually scanned through the timeseries plots of the remaining sensors and threw away the ones which were clearly garbage (those reporting a perfectly linear nitrate concentration over their entire lifespan, for instance). This left use with 85 sensors.

From these nitrate timeseries, we define two targets: a regression target given by the daily max nitrate value observed by a sensor and a classification target given by the violation category of a sensor. A "violation" is defined by the FDA as a nitrate concentration above 10 mg/L (though in reality levels far below this are still unsafe for human consumption), so in the latter case the target is "1" if the daily maximum nitrate level rose above 10 mg/L and is "0" otherwise.

Sensor data came from two different sources depending on whether the sensors were state or federally operated:
- IWQIS (Iowa Water-Quality Information Network) sensors: Jerry from the IWQIS gave us a 3.3 GB csv containing the sensor timeseries for 61/85 of our final sites.
- USGS NWIS (National Water Information Service) sensors: 24/85 final sensors come from federally operated sites, and were downloaded via API requests.


## Covariates

Water-borne nitrate comes from a variety of sources, but the primary source is agricultural. Many crops, notably corn, require copious nitrogen input into the soil. Excess nitrogen in the soil then leaches into the watershed, largely through precipitation, where some of it flows into the path of a nitrate sensor. Given this, we chose the following as our primary covariates:

- *Crop Distribution*, calculated yearly via satellite imagery from 2000 - 2008. Accessed using the USDA Crops Data Layer to produce GeoTIFFs, where each pixel corresponds to one of 254 different land-use types (Corn, Soybeans, NonAgg, etc) at a resolution of 30m x 30m. We refer to this as the *crop data*.
- *Nitrogen Surplus 2000-2017*, a model of yearly average nitrogen surplus in the continental United States from 1930-2017 calculated via the gTREND model in the 2026 Nature Paper [gTREND-Nitrogen - Long-term nitrogen mass balance data for the contiguous United States (1930-2017)](https://www.nature.com/articles/s41597-026-06576-x). One massive GeoTIFF downloaded manually for each year. We refer to this as the *surplus data*.
- *Daily Historical Weather*, precipitation in inches from IEM and then min/max temperature in Celsius, min/max humidity, vapor pressure difference, evapotransiration in (mm), solar radiation in Joules and 1000h fuel moisture from gridMET. We refer to this as the *weather data*.

Other features we considered but decided against:
- *SSURGO*, detailed static information about soil across the United States. We considered using it as a source of static categorical information for the land surrounding each sensor, but the data was hard to organize and preliminiary EDA demonstrated it had little effect on model performance.
- *OpenET Database*, a wonderful dataset with detailed evapotranspiration data. We already had rudimentary evapotranspiration data from gridMET, and felt the difficulty of incorporating this additional datastream outweighed the potential accuracy boost it might provide.
- *Point source pollutors*. In addition to agricultural contaminants, there exist discrete point sources of nitrogen pollution, slaughterhouses or pig farms for instance, which provide a roughly constant stream of nitrogen into certain rivers in Iowa. We couldn't find a good single datasource documenting this data in the data-scraping phase of this project, but think the inclusion of these features would provide the single-best model improvement.
- *USDA LTAR*: The Long Term Agricultural Network is a conglomerate of 19 different USDA-affiliated research stations studying various agricultural questions. In theory their data is publically accessible, in practice, each site provides its own mechanism for searching and accessing their data. There are likely incredibly useful features hiding somewhere here, but we weren't able to find any.
- *NASA SMAP*: This provides soil moisture data updated every 2-3 days at 9km resolution. We didn't bother examining it in EDA. When we began training in earnest we were surprised to find that the inclusion of 1000h fuel moisture, a tag-along fire-hazard indicator in our weather data, measurably improved our models. We reason it is a proxy for long-term hydrology of the soil, which affects the passage of nitrogen from soil to water. While we discovered this too late to incorporate NASA SMAP in our final model, it could be a useful thing to add in a future version.
- *USDA NASS Quickstats*: A massive collection of agricultural data including yield, harvested area, production volume, fertilizer sales and irrigated area. Hard to access, much of it is contained in pdf reports, and potentially of limited usefulness due to its unpredictable collection frequency. An ambitious data scraper could likely find useful categorical labels and fertilizer proxies in here, though.
- *Additional live sensor features*: The sensors themselves reported other features besides nitrate, things like pH, oxygenation level and flow rate. Which particular features were reported varied widely across the various sites, but many of these features could be useful if monitoring locations were filtered to ensure the feature of interest was present in the data for all sites.

## Oragnization of the Data

Two peculiarities drove our data organization. First, the only pollutants which can possibly contribute to the reading of nitrate sensors deployed in water are those directly upstream of the sensor. This means that the area relevant to a given sensor is the *drainage basin* of the sensor, defined as the set of points directly upstream from the sensor. Second, our crop, surplus and weather data exist on rectangular grids at resolutions of 30m, 250m and 4km respectively. Part of our data cleaning necessarily included, therefore, the calculation of the drainage basins for each sensor and the reconciliation of these three different grids (details on this in the following two subsections). After this process we had, associated to each of our 85 deployed monitoring sites,

- a *basin* parquet, encoding the geometry of the sensor's drainage basin geometry
- a *grid* parquet, data concerning the portion of the weather grid falling inside the sensor's basin, used for joining crop, surplus and weather datasets within a site as well as joining individual basin datasets together
- a *crop* parquet containing all crop data inside a site's basin aggregated to the weather grid
- a *surplus* parquet containing all surplus data inside a site's basin aggregated to the weather grid
- a *weather* parquet containing a daily timeseries of all weather data from the start to end of a sensor's lifetime, buffered on either end by 2 months and aligned to the weather grid natively
- a *water* parquet containing the 5-15 minute frequency nitrate concentration timeseries along with other quantities inconsistently tracked across sites.

### Basin Calculation

Irrelevant/relevant area erroneously included/discluded from a drainage basin could severly limit model performance, hence every other piece of this project was downstream of the drainage basin calculation (get it?). We spent a fair amount of time on this step as we wanted to ensure the ceiling for our model's capabilities was as high as possible. There were three different methods we used to find these drainage basins, and each had its own failure mode:

1. *snap to reference*: use a USGS API to lookup precomputed drainage basins by snapping the GPS coordinates of a sensor to the nearest reference point. This worked well for most sites, but failed horribly for sensors place on small streams near intersections with major rivers: in the worst cases this led to the drainage basin of a small local sensor incorrectly including all of Montana.
2. *authority lookup*: use the unique identifier of a site (its so called `site_uid`) to lookup the drainage basin directly from either a federal or state authority. This worked quite well for every USGS site but quite badly for the IWQIS sites.
3. *compute the basin algorithmically*: the IWQIS monitoring site provides a feature for displaying a drainage basin for an arbitrary pin drop on the map, and it almost works. It runs client-side in Javascript, so we used Claude to scrape through the site source and reconstruct the algorithm (it uses a GeoTIFF aligned to a reference grid with the direction of flow at every cell rounded to one of the 8 cardinal directions, and then finds the boundary of a basin via depth-first-search).

We then went through and manually selected a basin for each site using one of these three methods, defaulting to 1 and 2 and only resorting to 3 when the other two were nonsensical. After this process there were *still* four sites with basins which included large amounts of downstream area or failed to include the sensor location itself. These four sites were all state operated, so we used either 1 or 3 to calculate basins for pins dropped near the offending locations and then chose those basins which seemed most reasonable and/or most closely matched the one displayed by IWQIS. At the end of this process we doubted the supposed irrefutability of the IWQIS's basins, and think that ours might be marginally more accurate.

The manually chosen preference basins were locked in to a read-only csv and not changed over the course of the project.

### Grid Reconciliation

The weather grid was far larger than either the surplus or crop grids, so the obvious approach here was to snap surplus and crop cells to the nearest weather cell and then aggregate. Unfortunately, the weather grid didn't align to latitude and longitude lines, it was slightly curved, meaning every edge partitioned the cells of the other two grids. This didn't matter for the crop grid (30m/4km resolution ratios), but could have lead to nontrivial error in the surplus aggregation.

We first calculated the Voronoi cells for the weather reporting locations. Then we aggregated the surplus data to the weather grid weighted by the area of intersection of surplus grid cells and the weather Voronoi cells, possible using the GeoPandas and Shapely python packages.

The aggregation was easier for crops, we simply summed the total number of pixels of each CDL category in each weather grid cell. To avoid keeping all 254 categories, many of which were redundant (Corn, Sweet Corn, and Pop Corn are all separate categories) we applied an intermediate filter to combine categories. Our final crop category list was "Corn", "Soybeans", "HayPasture" (things like Alfafa), "Small Grains", "Fallow", "Non Agg" (catch all for ~0 Nitrogen contributors) and "Other" for all CDL categories not directly addressed.

At the end of this we had a global weather grid parquet with corresponding grid-aggregated crop and surplus files. We additionally included one grid file per monitoring location containing the portion of the weather grid covering the site's drainage basin. Each of these site-grid files included the fraction of area of each cell contained in the basin as well as an estimate of the distnace from the centroid of each cell to the sensor, calculated using an algorithm adapted from the resources in basin-calculation-method (3) above.

## EDA

Preet and Erin and Jay and Xiaoying please write this

!-----NOT WRITTEN-----!

### Findings during EDA

!-----NOT WRITTEN-----!
- seasonality
- crop good predictor
- distance from sensor matters
- high autocorrelation, "predict yesterday" baseline really good
- classification way better than regression, can beat out "predict yesterday" with just covariates, good candidate for virtual sensor

## Cross-Validation

We perform two different kinds of cross validation: "leave one site out" (LOSO), in which a single site is left out of training at a time, and "leave one family out" (LOFO), in which an entire family of basins is left out of training at a time. The reason for this is as follows.

There are two ways that two basins can overlap: *spatially* if one basin is physically contained in another and *temporally* if the two sites record observations simultaneously. In LOSO, if the left-out site happens to overlap spatially *and* temporally with another site in the training set, then there is data leakage. The solution to this is to build a "conflict graph" in which two sites are connected by an edge if and only if one basin is contained in the other and their observation periods overlap and then ensure that connected components of this graph are never partitioned across the training/test split. This is LOFO. LOFO is stricter than LOSO, and in practice, it is *too* strict. Our EDA demonstrated that the cells immediately surrounding a sensor are vastly more important than the cells in the outer reaches of the basin, meaning that severe data leakage only actually occurs when two sensors are right next to each other. Most conflict edges result from one mega-basin containing many smaller basins associated to sensors monitoring streams flowing into a major river. The cells affecting the readings for these smaller basins have essentially no effect on the mega-basin nitrate readings.

### Findings during CV Experiments

- seasonality well accounted for by sin/cos signal
- lat/lon matters a LOT (likely a good proxy for other hidden geographic covariates not present in dataset, point pollution sources for instance)
- fuel 1000h moisture super useful for some reason
- 

!-----NOT WRITTEN-----!

## Final model

!-----NOT WRITTEN-----!

## Future work

The data suite we've created makes it possible both to incorporate new features as they become available and to rapidly prototype/develop new models for specific use cases. To that end, here are some ideas we have for other models that could be built using the data we've assembled:

1. **Flaky sensor enhancement**. The models we built were trained on the 85 "best" sensors available to us (see [the targets section](#target)). Our regression results improve significantly when trained with access to historical site data. One these models could be used to supplement missing measurements from the other 162 - 85 sensors.
2. **Spike prediction**.