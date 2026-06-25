"""Orchestrate the weather pipeline (called by data/make_data.py).

Runs the three builders in dependency order:
    1. make_grid           — per-site grids + global_grid.parquet (weather_grid/)
    2. make_global_weather — IEM precip + gridMET, one file per year (weather_global/)
    3. make_basin_weather  — per-site weather sliced from the global files (weather_data/)
"""

import argparse

import make_grid
import make_global_weather
import make_basin_weather


def main(api_keys=None, force: bool = False):
    print("── weather/make_grid ──")
    make_grid.main(api_keys, force=force)
    print("── weather/make_global_weather ──")
    make_global_weather.main(api_keys, force=force)
    print("── weather/make_basin_weather ──")
    make_basin_weather.main(api_keys, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild everything.")
    args = parser.parse_args()
    main(force=args.force)
