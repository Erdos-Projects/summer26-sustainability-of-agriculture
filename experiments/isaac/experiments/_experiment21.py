"""_experiment21 -- can the model pick out the worst ungauged basins? (regression)

exp20 lifted the between-site metrics (REG between_r2 0.2247 -> 0.3514, CLF between_rate_r2 0.4139 -> 0.4883), which is the axis a sensor-siting or triage question rides on: rank the unmonitored basins, investigate the top of the list. between_r2 does not answer that question. It is an R^2 over site means, so it is dominated by the bulk, and a boosted ensemble shrinks extremes toward the mean -- it can score respectably while flattening precisely the sites a shortlist is made of.

Three numbers, on LOFO out-of-fold predictions so every site is scored as if ungauged. The first two are what cook._tail_rank records at TAIL_FRAC for the score tables; the third is computed here only.

    site_ap      average precision for "in the worst `frac` of sites". Chance is `frac` itself.
    captured     share of the achievable excess a shortlist of that size actually finds, in mg/L terms.
    tail_slope   slope of actual on predicted across the flagged sites -- 1.0 is honest spread, ~0 is a flattened blob. DIAGNOSTIC ONLY, and only legible ACROSS the ladder: on the 79-site REG cohort it ran 1.120 / 0.517 / 1.075 at frac 0.1 / 0.25 / 0.5, non-monotone, because it fits a line through as few as 8 points. Read it when site_ap disagrees between a narrow and a wide frac -- a genuinely flattened tail shows up as a slope well under 1 at the narrow end. Never quote one value alone.

REPORTED OVER A LADDER OF FRACS, and the ladder is the point. At frac=0.5 the question is "which half of the state is the problem half" -- well powered, but NOT tail-sensitive: measured on a synthetic model whose top 15% is deliberately compressed, site_ap and captured both rate it ABOVE a uniformly-good model (0.995 vs 0.955, 0.991 vs 0.879), because the compression happens inside the top half where a half-split cannot resolve it. The same pair separates them 2.7x the right way at frac=0.1. Read 0.5 for screening and 0.1 for outliers; if they disagree, the tail is being flattened.

BOOTSTRAP RESAMPLES FAMILIES, NOT SITES. Sites in one basin family share upstream area and are the reason LOFO exists; resampling sites would treat them as independent and give an interval that is too narrow.

TWO THINGS THIS CANNOT MEASURE, both of which belong beside any claim built on it:

  * SELECTION BIAS. The cohort's sensors were not sited at random -- they sit where problems were expected or where access was practical. "Which held-out SENSOR site is worst" is therefore an easier question than "which of 16,760 reaches is worst", because every site here already passed a siting filter no ungauged reach has. These are an UPPER BOUND on field performance.
  * The real cost asymmetry. A false positive costs a site visit; a false negative leaves a bad basin unmonitored. Every metric here weights them equally and reality does not.

Run:  python experiments/isaac/experiments/_experiment21.py [--full]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.access import get_site_ids  # noqa: E402
from src.eval.cook import _features, _grouped_oof, _pool, _tail_rank, _target, basin_groups  # noqa: E402
from src.features.features import daily_nitrate  # noqa: E402
from src.features.recipes import light_CLF, light_REG  # noqa: E402
from src.models.train import xgb_for  # noqa: E402

MIN_OBS = 500
FRACS = (0.10, 0.25, 0.50)
N_BOOT = 2000
SEED = 0


def site_table(y, oof, site, family) -> pd.DataFrame:
    """Per-site actual and predicted means, plus each site's basin family -- the table every metric reads."""
    ok = ~(np.isnan(y) | np.isnan(oof))
    tab = pd.DataFrame({"y": y[ok], "p": oof[ok], "g": site[ok], "fam": family[ok]})
    out = tab.groupby("g")[["y", "p"]].mean()
    out["fam"] = tab.groupby("g")["fam"].first()
    return out


def tail_slope(sm: pd.DataFrame, frac: float) -> float:
    """Slope of actual on predicted across the PREDICTED worst k -- the shrinkage diagnostic cook deliberately does not record."""
    n = len(sm)
    k = max(1, int(round(frac * n)))
    y, p = sm["y"].to_numpy(float), sm["p"].to_numpy(float)
    sel = np.argsort(-p)[:k]
    return float(np.polyfit(p[sel], y[sel], 1)[0]) if k > 1 and np.ptp(p[sel]) > 0 else float("nan")


def _boot(sm: pd.DataFrame, frac: float, n_boot=N_BOOT, seed=SEED) -> dict[str, tuple[float, float]]:
    """Percentile CIs for the trio, resampling BASIN FAMILIES with replacement.

    The family is the independent unit here -- that is the premise of LOFO -- so a site-level resample would understate the spread. A family drawn twice contributes its sites twice, on a fresh index, so duplicates stay distinct instead of collapsing.
    """
    rng = np.random.default_rng(seed)
    fams = sm["fam"].unique()
    by_fam = {f: sm[sm["fam"] == f] for f in fams}
    acc: dict[str, list[float]] = {}
    for _ in range(n_boot):
        draw = pd.concat([by_fam[f] for f in rng.choice(fams, len(fams), replace=True)], ignore_index=True)
        for k, v in _tail_rank(draw, frac=frac).items():
            if not np.isnan(v):
                acc.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items() if v}


def report(sm: pd.DataFrame, label: str, units: str) -> dict:
    """Print the ladder for one task and return it flat."""
    n, n_fam = len(sm), sm["fam"].nunique()
    cohort = sm["y"].mean()
    print(f"\n── {label}: {n} held-out sites in {n_fam} basin families, cohort mean {cohort:.3f} {units}")
    print(f"{'frac':>6} {'k':>4} {'site_ap':>9} {'(chance)':>9} {'captured':>10} {'tail_slope':>11}   95% CI on site_ap")
    row = {"n_sites": n, "n_families": n_fam, "cohort_mean": float(cohort)}
    for frac in FRACS:
        m = dict(_tail_rank(sm, frac=frac), tail_slope=tail_slope(sm, frac))
        ci = _boot(sm, frac)
        k = max(1, int(round(frac * n)))
        lo, hi = ci.get("site_ap", (float("nan"),) * 2)
        print(f"{frac:>6.0%} {k:>4} {m['site_ap']:>9.3f} {frac:>9.2f} {m['captured']:>10.3f} "
              f"{m['tail_slope']:>11.3f}   [{lo:.2f}, {hi:.2f}]")
        row.update({f"{key}@{int(frac * 100)}": v for key, v in m.items()})
        row.update({f"site_ap@{int(frac * 100)}_lo": lo, f"site_ap@{int(frac * 100)}_hi": hi})

    # What a shortlist would actually have found, in the target's own units.
    k = max(1, int(round(0.10 * n)))
    best = sm["y"].nlargest(k).mean()
    got = sm.loc[sm["p"].nlargest(k).index, "y"].mean()
    print(f"\n  a top-{k} shortlist: picks basins averaging {got:.3f} {units}, "
          f"against {best:.3f} achievable and {cohort:.3f} at random")
    row.update({"shortlist_k": k, "shortlist_got": float(got), "shortlist_best": float(best)})
    return row


def main(task="reg", full=False):
    recipe, target = (light_REG, "nitrate_con") if task == "reg" else (light_CLF, "violation")
    sites = [s for s in get_site_ids() if daily_nitrate(s).dropna().shape[0] >= MIN_OBS]
    if not full:
        sites = sites[:20]
    print(f"[{task}] pooling {len(sites)} sites ...")
    pool = _pool(recipe, sites, target, min_rows=500, progress_label=f"exp21 {task}")

    # X/y exactly as cook_many builds them -- _target int-casts for clf, which XGBClassifier needs.
    feat = _features(pool, target)
    X, y = pool[feat], _target(pool, target, task)
    # _pool tags the site column "site", not "site_uid" -- cook_many reads pool["site"] too.
    groups = basin_groups(pool["site"])
    # LOFO through cook's own path, so these sit on the same footing as lofo_between_r2: GroupKFold over
    # basin families, every site predicted by a model that never saw its family.
    oof = _grouped_oof(X, y, groups, task, n_splits=5, **xgb_for(recipe, task))

    sm = site_table(y.to_numpy(float), oof, pool["site"].to_numpy(), groups.to_numpy())
    label, units = ("REG (site-mean nitrate)", "mg/L") if task == "reg" else ("CLF (site violation rate)", "rate")
    row = report(sm, label, units)

    out = Path(__file__).parent / "test_results"
    out.mkdir(exist_ok=True)
    stem = Path(__file__).stem + ("" if task == "reg" else "c")
    sm.to_csv(out / f"{stem}_site_means.csv")
    pd.DataFrame([row]).to_csv(out / f"{stem}.csv", index=False)
    print(f"\nwrote {stem}.csv and {stem}_site_means.csv to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the whole filtered cohort rather than the first 20 sites")
    ap.add_argument("--task", default="reg", choices=("reg", "clf"))
    a = ap.parse_args()
    main(task=a.task, full=a.full)
