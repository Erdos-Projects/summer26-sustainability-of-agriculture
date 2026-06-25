"""Orchestrates the creation/initialization of the data."""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

DATA_DIR = Path(__file__).parent
SCRIPTS = [
    "water/make_water.py",
    "map_overlays/make_map_overlays.py",
    "basins/make_basins.py",
    "weather/make_weather.py",    # builds the shared grid + weather that surplus + crops aggregate onto
    "surplus/make_surplus.py",
    "crops/make_crops.py",        # depends on the grid; independent of surplus
    "aux/make_aux.py",            # basin-containment graph; needs the grids + basins
]


def get_api_keys():
    import tomllib

    API_KEYS = "api-keys.toml"

    with open(DATA_DIR / API_KEYS, "rb") as f:
        keys = tomllib.load(f)

    return keys


def load_and_run(script, force: bool = False):
    # import the script properly as a module -------------------------------------------------
    spec = importlib.util.spec_from_file_location(f"dataset_{script.parent.name}", script)
    module = importlib.util.module_from_spec(spec)

    # change directory and add script's dir to sys.path so sibling imports work -------------
    cwd = os.getcwd()
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))

    # run the script -------------------------------------------------------------------------
    try:
        spec.loader.exec_module(module)  # run top-level code like global var definitions
        module.main(get_api_keys(), force=force)  # execute the method main()
    finally:
        os.chdir(cwd)
        sys.path.remove(str(script.parent))


def main(force: bool = False):
    for name in SCRIPTS:
        script = DATA_DIR / name
        if not script.exists():
            raise FileNotFoundError(f"Expected {script}")

        print(f"\n=== {script.parent.name + "/" + script.name} ===")
        load_and_run(script, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite existing parquets.")
    args = parser.parse_args()
    main(force=args.force)
