# Guide to data

## IWQIS (Iowa Water Quality Information Systems)

This is the data you can see live [here](https://iwqis.iowawis.org/) and was given to us by Jerry. It integrates data gathered by IIHR (University of Iowa's Hydroscience and Engineering center) and the USGS (United States Geological Survey). It offers nutrient data with a range of water-related information such as precipitation, stream flow, and soil moisture.

The complete data forms a 3.3 GB CSV. This is the unaggregated raw data from all sites in the study going back to 2012. Measurements are made in 5-15 minute intervals. To save space and time, **don't push the uncompressed CSV to the github repo,** use the provided script to rebuild it from chunks.

### Access
Use `reassemble.csv` to reassemble the chunked files from `IWQIS/chunks` into a single file with the command

`python reassemble.py chunks/iwqis_alldata_manifest.json --output fulldata.csv`

### List of files
- `chunks/`: contains the chunked data for easier transimission.
- `fulldata.csv`: the main dataset. Contains all unaggregated data going back to the start of the study in 2012, observations made in 5-15 minute increments.
- `measures.csv`: metadata on measurements.
- `params.csv`: metadata on parameters.
- `site.csv`: metadata on the observation sites.
- `split_csv.py`: utility script for producing the chunked data in `chunks/` from the main dataset.
- `reassemble.py`: utility script for reassembling the full dataset from the output of `split_csv.py`.
- `verify_data.py`: compare two versions of the main data.
