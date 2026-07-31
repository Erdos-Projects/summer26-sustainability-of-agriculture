"""_experiment20 -- does long-run basin composition help, and does the sd block earn its place? (regression)

Every covariate joins per YEAR, so the model only ever sees one year's land cover -- a snapshot carrying that year's weather, rotation phase and price response. features.longrun_from_blocks reduces those per-year shares over features.LONGRUN_YEARS (the training span, 2008-2026, intersected with what each block holds: crops to 2025, surplus to 2017), giving one scalar per site per column. Being constant within a site it cannot move within_r2 at all -- this is aimed at the BETWEEN-site metrics, which are this repo's weak axis.

  base       the shipped light_REG, long-run columns removed
  mean       + pct_<class>_mean_b{k}, surplus_kgha_norm_mean_b{k}
  mean_sd    + the _sd_b{k} block on top

Two reasons this is re-measured here rather than inherited from the sibling repo's exps 32/32c (which found REG mean +0.0420 lofo_r2 with between_r2 0.044 -> 0.166, and the sd block COSTING -0.0149): its ring geometry was 2 buckets at 50 km against this repo's (5k, 50k), and it reduced over the full CDL record with no training-window floor -- and the arm most exposed to that is exactly `sd`. See the LONGRUN_YEARS comment in features.py for why the floor matters.

METHOD. All three arms score ONE cached frame per site, masked by column, so the arms cannot drift apart in how a block was computed -- the same guarantee the sibling's kind="masked" gives. The xgb config is held fixed at the deployed light_REG tuning: the arms differ in width and a config tuned for one width would bias the others, so this ranks the arms and the retune afterwards sets the shipped number.

Run:  python experiments/isaac/experiments/_experiment20.py [--full]
"""

import argparse
import sys
import warnings
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.access import get_site_ids  # noqa: E402
from src.eval.cook import compare_many, save_comparison  # noqa: E402
from src.features.features import daily_nitrate  # noqa: E402
from src.features import recipes  # noqa: E402
from src.features.features import LONGRUN_STAT_NAMES  # noqa: E402
from src.features.recipes import light_REG, light_CLF  # noqa: E402
from src.models.train import xgb_for  # noqa: E402

MIN_OBS = 500  # 79 of 83 sites, matching cook_many's own min_rows default rather than the >=1500 the older experiments in this directory use -- more basin families is what LOFO is short of


def _longrun_cols(df, stat):
    return [c for c in df.columns if f"_{stat}_b" in c or c.endswith(f"_{stat}")]


def arms(recipe):
    """{arm: recipe} over one lru_cached base frame per site -- the arms are column masks, never separate builds.

    The base frame is built with EVERY stat, whatever the shipped constants say. Masking can only remove what the recipe emitted, so once LONGRUN_STATS has been narrowed to ("mean",) the mean and mean_sd arms would otherwise be the same frame and the run would report a comparison it never made -- quietly, since both arms still produce plausible numbers. Overriding the constant costs nothing: longrun_from_blocks always computes both stats and _agg_block caches them, so this changes only which ones get broadcast.
    """

    @lru_cache(maxsize=None)
    def base(site):
        live = (recipes.LONGRUN_STATS, recipes.LIGHT_LONGRUN_STATS)
        saved = [(d, dict(d)) for d in live]
        for d in live:
            d.update({k: LONGRUN_STAT_NAMES for k in d})
        try:
            return recipe(site)
        finally:
            for d, old in saved:
                d.clear()
                d.update(old)

    def mask(drop_stats):
        def fn(site):
            df = base(site)
            drop = [c for s in drop_stats for c in _longrun_cols(df, s)]
            return df.drop(columns=drop)

        return fn

    return {"base": mask(("mean", "sd")), "mean": mask(("sd",)), "mean_sd": mask(())}


def main(task="reg", sites=None, full=False, extra=False):
    recipe, target, name = (light_REG, "nitrate_con", "reg") if task == "reg" else (light_CLF, "violation", "clf")
    if sites is None:
        sites = [s for s in get_site_ids() if daily_nitrate(s).dropna().shape[0] >= MIN_OBS]
        if not full:
            sites = sites[:12]
    r = arms(recipe)
    print(f"[{name}] {len(r)} arms x {len(sites)} sites")
    res = compare_many(r, sites, target_col=target, task=task, extra_importance_test=extra, **xgb_for(recipe, task))
    cols = [c for c in res.columns if c.startswith("lofo_") or c.startswith("loso_") or c == "n_rows"]
    print(res[cols].round(4).to_string())
    Path(__file__).parent.joinpath("test_results").mkdir(exist_ok=True)
    stem = Path(__file__).stem + ("" if task == "reg" else "c")
    save_comparison(res, str(Path(__file__).parent / "test_results" / f"{stem}.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the whole filtered cohort rather than the first 12 sites")
    ap.add_argument("--task", default="reg", choices=("reg", "clf"))
    ap.add_argument("--extra", action="store_true", help="also run permutation importance (slow)")
    a = ap.parse_args()
    main(task=a.task, full=a.full, extra=a.extra)
