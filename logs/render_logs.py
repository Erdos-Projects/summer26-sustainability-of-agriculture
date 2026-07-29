"""Render logs/fulltrain_logs.json -> human-readable markdown.

Run:  python logs/render_logs.py      (paths resolve relative to this file, so any cwd works)

Writes two files next to the JSON log:
  scores.md       -- one headline block per model run: recipe, feature list, score table
  importances.md  -- one block per run: gain + permutation importance table (the heavier data)

The log is a dict keyed by sequential integer (see src/models/train.log_metadata); each entry has
name / recipe / features / target_col / task / xgb / score / importance / importance_perm.
"""

import json
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LOGFILE = _HERE / "fulltrain_logs.json"
_SCORES_MD = _HERE / "scores.md"
_IMPORTANCES_MD = _HERE / "importances.md"


def _model_type(entry):
    """'reg'/'clf' task -> 'REG'/'CLF' (falls back to the upper-cased task string)."""
    return {"reg": "REG", "clf": "CLF"}.get(entry.get("task", ""), str(entry.get("task", "?")).upper())


def _fmt(v):
    """Compact table cell: ints as-is, floats to 4dp (NaN -> 'NaN'), else str()."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v == int(v) and abs(v) < 1e15:  # count-like (pandas upcasts n_sites/n_rows to float)
            return str(int(v))
        return f"{v:.4f}"
    return str(v)


def _make_markdown_table(df, index_label=None):
    """Render `df` as a markdown table, cells passed through `_fmt`. With `index_label`, the row
    index is emitted as the leading column under that header (the transposed importance table uses
    it for the Gain/Perm row labels)."""
    header = ([index_label] if index_label is not None else []) + [str(c) for c in df.columns]
    lines = [header, ["---"] * len(header)]
    for idx, row in df.iterrows():
        lines.append(([str(idx)] if index_label is not None else []) + [_fmt(v) for v in row])
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _score_table(score):
    # transposed: metric names across the header, a single row of values
    return _make_markdown_table(pd.DataFrame([score]))


def beta_table_md(entry):
    """The decision-threshold table for a classifier run, as a markdown string ('' if absent).

    Reads the `beta_table` src/models/train.py logs from the CV's honest LOFO out-of-fold vector.
    Columns beyond recall/precision/fdr are DERIVED here rather than stored, because they follow
    exactly from (base_rate p, recall r, precision pi):

        TP = r*p        FP = TP*(1-pi)/pi       FN = p - TP        TN = (1-p) - FP

    so accuracy = TP+TN and FPR = FP/(1-p). Accuracy is included with the do-nothing baseline
    beside it, because at a ~26% base rate "never alarm" scores ~74% and every operating point
    above beta 1 is WORSE than that -- which reads as a broken model unless the baseline is shown.

    Returns only classification runs' tables; regression entries and pre-existing runs logged
    before this field existed both yield ''.
    """
    table = entry.get("beta_table")
    if not table:
        return ""
    p = entry.get("base_rate")
    if p is None:
        return ""

    rows = []
    for r in table:
        rec, pi = r["recall"], r["precision"]
        tp = rec * p
        fp = tp * (1 - pi) / pi if pi > 0 else float("nan")
        acc = tp + ((1 - p) - fp)
        rows.append(
            {
                # pre-formatted: _fmt renders int-valued floats as ints, which would print the grid as
                # "0.5000, 1, 1.5000, 2" -- ragged for a column whose values are all half-steps.
                "beta": f"{r['beta']:.1f}",
                "tau": r["tau"],
                "recall": rec,
                "fdr": r["fdr"],
                "precision": pi,
                "accuracy": acc,
                "fpr": fp / (1 - p) if p < 1 else float("nan"),
                "lift": pi / p if p > 0 else float("nan"),
            }
        )

    cover = entry.get("beta_table_coverage")
    note = (
        f"Base rate {p:.4f} (in the scored rows) — 'never alarm' is {1 - p:.1%} accurate. "
        f"Lift is precision ÷ base rate at that point."
    )
    if cover is not None and cover < 0.999:
        note += f" **Scored on {cover:.1%} of pooled rows** — a capped true-LOFO run, so this describes only the eligible families."
    return note + "\n\n" + _make_markdown_table(pd.DataFrame(rows))


def _scores_block(key, entry):
    feats = ", ".join(entry.get("features", [])) or "_(none logged)_"

    h_pct = entry.get("max_holdout_pct", "")
    if h_pct != "":
        h_pct = f"(max_holdout_pct = {h_pct})"

    return (
        f"# {_model_type(entry)} Model {key}: {entry.get('name', '')}\n\n"
        f"**Recipe:** {entry.get('recipe', '?')}  \n"  # trailing 2 spaces -> markdown hard line break
        f"**True Lofo:** {entry.get(f'true_lofo', '?')} {h_pct} \n"
        f"**Notes:** {entry.get('notes', 'None')} \n\n"
        f"**Features:** {feats}  \n"
        f"**Scores:**\n\n"
        f"{_score_table(entry.get('score', {}))}\n\n"
        f"**Beta Table:**\n\n"
        f"{beta_table_md(entry)}"
    )


def _importance_table(entry):
    # access the gain of the entry dict safely
    # falls back to {} when importance key exists but is None
    gain = entry.get("importance", {}) or {}
    perm = entry.get("importance_perm", {}) or {}
    if not gain and not perm:
        return "_(no importances logged)_"

    feats = list(gain) + [f for f in perm if f not in gain]  # gain is already gain-ranked
    d = {
        "gain": [gain[f] if f in gain else "-" for f in feats],
        "perm": [perm[f] if f in perm else "-" for f in feats],
    }
    # transposed: features across the header, a Gain row and a Perm row (missing -> em dash)
    df = pd.DataFrame(d, index=feats).sort_values(by="perm", ascending=False)

    return _make_markdown_table(df, index_label="features")


def _importances_block(key, entry):
    return (
        f"# {_model_type(entry)} Model {key}\n\n"
        f"Recipe: {entry.get('recipe', '?')}\n"
        f"Date: {entry.get('timestamp', '?')}\n"
        f"{_importance_table(entry)}\n"
    )


def render():
    if not _LOGFILE.exists() or _LOGFILE.stat().st_size == 0:
        raise SystemExit(f"no training log at {_LOGFILE} -- run a training build first")
    with open(_LOGFILE) as f:
        log = json.load(f)
    keys = sorted(log, key=int, reverse=True)  # integer order despite JSON's string keys

    _SCORES_MD.write_text("\n---\n\n".join(_scores_block(k, log[k]) for k in keys) + "\n")
    _IMPORTANCES_MD.write_text("\n---\n\n".join(_importances_block(k, log[k]) for k in keys) + "\n")
    print(f"wrote {_SCORES_MD}  ({len(keys)} models)")
    print(f"wrote {_IMPORTANCES_MD}  ({len(keys)} models)")


if __name__ == "__main__":
    render()
