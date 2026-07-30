# Links

## Code

- **[Project repository](https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture)** — everything behind this site: the data pipeline, the models, and the widget itself.
- **[Ongoing work](https://github.com/ikmartin/sustag)** — the fork where development continues, aimed at deploying heavier models on a dedicated server incorporating live in-situ nitrate data across the continental United States. 

## This tool

- **[Live widget](https://erdos-projects.github.io/summer26-sustainability-of-agriculture/)** — the page you are on, if you want to send someone the link.
- **[Presentation video](https://www.youtube.com/watch?v=O_ZCylQCXe8)** — five minutes on the problem and the approach. Also embedded under *Presentation*.

## Data sources

- **[IWQIS](https://iwqis.iowawis.org/)** — Iowa Water Quality Information System, source of the `WQS*` nitrate sensors and the inspiration for this widget's design.
- **[USGS NLDI](https://api.water.usgs.gov/nldi/linked-data)** — the hydrologic network navigation service behind every drainage basin here, and [NWIS](https://api.waterdata.usgs.gov/) for the `USGS-*` gauges.
- **[gridMET](https://www.climatologylab.org/gridmet.html)** — daily gridded weather.
- **[USDA Cropland Data Layer](https://nassgeodata.gmu.edu/CropScape/)** — annual 30 m crop classification.
- **[gTREND](https://www.nature.com/articles/s41597-026-06576-x)** — the long-term nitrogen mass balance dataset providing the soil surplus layer.

Full access details for every source, including which builder script consumes it, are under *Data inventory*.