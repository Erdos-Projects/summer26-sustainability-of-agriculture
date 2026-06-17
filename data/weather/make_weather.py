import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_rain as rain


def main(api_key):
    rain.main(api_key, site_uids=None)


if __name__ == "__main__":
    main()
