"""_experiment21c -- can the model pick out the worst ungauged basins? (classification)

The CLF half of _experiment21; see that file for the hypothesis, the three metrics, the frac ladder and the two things they cannot measure.

The per-site quantity here is the VIOLATION RATE -- the share of a site's days at or above 10 mg/L -- rather than mean concentration, so `captured` reads in rate units. That is the closer match to the siting question: an agency prioritising a gauge cares how often a basin is over the limit, not what its annual average is. It is also the better-measured axis, since exp20's CLF arm ran on 79 sites in 20 families against REG's 45 in 13.

Run:  python experiments/isaac/experiments/_experiment21c.py [--full]
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment21 import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the whole filtered cohort rather than the first 20 sites")
    a = ap.parse_args()
    main(task="clf", full=a.full)
