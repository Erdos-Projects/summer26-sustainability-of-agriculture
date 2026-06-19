"""Build crop data outputs from raw source files.

Inputs  (crops_raw/)
    ...

Outputs (crops_data/)
    ...

Usage
-----
    python make_crops.py           # process all, skip existing
    python make_crops.py --force   # rewrite all
"""

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "crops_data"
_META_DIR = _THIS_DIR / "crops_meta"
_RAW_DIR  = _THIS_DIR / "crops_raw"

sys.path.insert(0, str(_THIS_DIR.parents[1]))


def main(api_keys=None, force: bool = False) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _META_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite existing outputs.")
    args = parser.parse_args()
    main(force=args.force)
