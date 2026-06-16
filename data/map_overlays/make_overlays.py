import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_iowa_water_overlays as water_overlays


def main(api_key):
    water_overlays.main(api_key)


if __name__ == "__main__":
    main()
