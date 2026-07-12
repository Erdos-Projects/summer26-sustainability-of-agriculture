# Documentation for the `src` module

This module contains the major components of the codebase for this project. Here is its structure.

- `build`: contains scripts for building the data from source or for redownloading the data. Run `python make_data.py` once upon a fresh clone and after downloading the raw data from [the download link](https://utexas.box.com/s/h4bjxgsuydcl7ya6gpwyiqepl477cdo7) and placing its contents in `src/data/raw`.
- `data/`: submodule for accessing the data. The actual data files also live in this directory.
- `eval/`: submodule for training and evaluating models. Consists mostly of one script, `cook.py`
- `features/`: submodule for generating features from the data. The feature sets and the methods for generating them are defined here.
- `models/`: train and tune full models. Models persist in `models/models/` for later loading.
- `splits/`: a single script, `conflict_graph.py`, lives here. Used for handling the train/test splits, made difficult by data leakage made possible by the geographical nature of the data.

## Build

1. Download raw data from https://utexas.box.com/s/h4bjxgsuydcl7ya6gpwyiqepl477cdo7.
2. Place the contents for the download in `src/data/raw/`.
3. Navigate to `src/build/`.
4. Run the command `python make_data.py`.

Once that completes, the project should be setup.

### Catastrophic rebuild

If the raw data download link doesn't work, `make_data.py` will not succeed. These are instructions for rebuilding weather, crop and surplus data in this event.

#### Rebuild of weather data

This should work automatically when `make_data.py` is called, it'll just be very slow. The script `make_water.py` will redownload the 

#### Rebuild of crop data

Crop data must be redownloaded from the USDA CDL. There is a provided utility for this, use

```python
python -m src.build.util.clip_crops --download
python -m src.build._make_crops --force 
```

This will take a while. The second command ensures stale files are overwritten. Once it completes, `make_data.py` should run.

#### Rebuild of surplus data

Navigate to [gTREND-Nitrogen - Long-term nitrogen mass balance data for the contiguous United States (1930-2017)](https://www.nature.com/articles/s41597-026-06576-x) and navigate to the "Data Records" section. Follow the [figshare link](https://springernature.figshare.com/articles/dataset/gTREND-Nitrogen_-_Long-term_nitrogen_mass_balance_data_for_the_contiguous_United_States_1930-2017_/29897375) and download the `Surplus.zip`. It should consist of tif files named `Surplus_N_{year}.tif`. Take the files corresponding to 2000-2017 and place them in `src/data/raw/surplus/tif/` (eg `src/data/raw/surplus/tif/Surplus_N_2000.tif`, `src/data/raw/surplus/tif/Surplus_N_2001.tif`, etc.).

Once those files are in place, run

```python
python -m src.build.util.build_source
```

to produce the necessary parquets.

## Data

## Eval

## Features

## Models

## Splits

