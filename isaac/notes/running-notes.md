## List of good water sites

Big basin sites
["WQS0066", "WQS0065", "USGS-05474500", "WQS0020", "USGS-05420500"]
Not good training data anyways, throwing away. Now filtered out manually in `make_water.py`.

## Widget Notes

### `info_panel.py`

Currently graphing code located here. To plot rain, rainfall is aggregated by date and then a mean is taken. This throws away all spatial content in the graph. You could imagine a more useful way to aggregate, for instance, a weighted sum based on distance from the grid point to the site location calculated as flow distance. Rainfall closer to the site should tend to have a bigger impact on the site, me-thinks.

This also is slow for graphing likely. Probably better to pre-aggregate all the data and then call it on demand.