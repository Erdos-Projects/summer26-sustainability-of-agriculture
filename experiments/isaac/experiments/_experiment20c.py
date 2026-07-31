"""_experiment20c -- long-run basin composition, mean vs mean+sd (classification).

The CLF half of _experiment20; see that file for the hypothesis, the arms and the masking method.

Reported on lofo_prauc, deliberately. The sibling repo's exp 32c adopted mean+sd for CLF on lofo_auc (+0.0139 mean, +0.0074 sd) -- lofo_prauc only became the CLF headline there at exp 33c, so this block has never been scored on average precision, which is the metric this repo treats as load-bearing. lofo_between_rate_r2 is the between-site number to read alongside it; on the sibling it moved only +0.0043, well under the noise floor its own design note set.

Run:  python experiments/isaac/experiments/_experiment20c.py [--full]
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment20 import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the whole filtered cohort rather than the first 12 sites")
    ap.add_argument("--extra", action="store_true", help="also run permutation importance (slow)")
    a = ap.parse_args()
    main(task="clf", full=a.full, extra=a.extra)
