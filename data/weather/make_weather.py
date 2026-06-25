"""Orchestrate the weather pipeline (called by data/make_data.py).

Runs the three builders in dependency order:
    1. make_grid           — per-site grids + global_grid.parquet (weather_grid/)
    2. make_global_weather — IEM precip + gridMET, one file per year (weather_global/)
    3. make_basin_weather  — per-site weather sliced from the global files (weather_data/)

The global build (step 2) re-fetches gridMET, so it is skipped when weather_data
already exists -- the per-site files are what downstream uses, and they don't need
the global files rebuilt. Pass --force to rebuild it anyway.
"""

import argparse

import make_grid
import make_global_weather
import make_basin_weather


def _weather_data_present() -> bool:
    """True if any per-site weather file (combined or split part) exists."""
    data_dir = make_basin_weather._DATA_DIR
    return data_dir.exists() and any(data_dir.glob("*_weather*.parquet"))


def main(api_keys=None, force: bool = False):
    print("── weather/make_grid ──")
    make_grid.main(api_keys, force=force)

    if force or not _weather_data_present():
        print("── weather/make_global_weather ──")
        make_global_weather.main(api_keys, force=force)
    else:
        print("── weather/make_global_weather ── skipped (weather_data present; --force to rebuild)")

    print("── weather/make_basin_weather ──")
    make_basin_weather.main(api_keys, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild everything.")
    args = parser.parse_args()
    main(force=args.force)
