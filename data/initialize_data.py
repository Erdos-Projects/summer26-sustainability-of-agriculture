import importlib.util
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent
SCRIPTS = ["water/make_data.py", "map_overlays/make_overlays.py"]


def get_api_keys():
    import tomllib

    API_KEYS = "api-keys.toml"

    with open(DATA_DIR / API_KEYS, "rb") as f:
        keys = tomllib.load(f)

    return keys


def load_and_run(script):
    # import the script properly as a module -------------------------------------------------
    spec = importlib.util.spec_from_file_location(f"dataset_{script.parent.name}", script)
    module = importlib.util.module_from_spec(spec)

    # change directory to presrve relative-path assumptions in the script --------------------
    cwd = os.getcwd()
    os.chdir(script.parent)

    # run the script -------------------------------------------------------------------------
    try:
        spec.loader.exec_module(module)  # run top-level code like global var definitions
        module.main(get_api_keys())  # execute the method main()
    finally:
        os.chdir(cwd)


def main():
    for name in SCRIPTS:
        script = DATA_DIR / name
        if not script.exists():
            raise FileNotFoundError(f"Expected {script}")

        print(f"=== {script.parent.name + "/" + script.name} ===")
        load_and_run(script)


if __name__ == "__main__":
    main()
