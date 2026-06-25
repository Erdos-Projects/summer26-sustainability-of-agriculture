"""Leakage-aware train/test splits over the basin network.

The hard problem with splitting these sites is that two basins can contaminate
each other's evaluation when they are dependent on *both* axes at once:

  * spatially  — one basin is nested in (or overlaps) another, so they share
    drainage area, weather forcing, surplus and crop cells; and
  * temporally — their nitrate records overlap in calendar time (or sit close
    enough that nitrate's months-long autocorrelation still links them).

A pair that is related on only one axis is a weak leak: spatial-only means
shared *static* structure but different weather realizations, temporal-only
means common-mode weather forcing. A pair related on *both* axes is effectively
a duplicate observation and must never straddle the train/test boundary.

This module builds a *conflict graph* whose edges are exactly the both-axes
pairs, takes its connected components as indivisible split groups, and assigns
whole groups to folds. That exploits the spatial-but-not-temporal case (such
pairs are *not* connected, so they may land on opposite sides) to recover more
independent test units than a pure spatial family holdout — while still
guaranteeing no hard leak crosses the boundary.

Spatial relatedness comes from the directed basin-containment graph
(data.aux.get_basin_graph); temporal relatedness from each site's nitrate date
range, padded by ``buffer`` to account for autocorrelation.

See also: data/aux/make_aux.py (build_basin_graph / get_basin_graph).
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.model_selection import GroupKFold

from .aux import get_basin_graph
from .water import get_site_ids, get_water

# Default autocorrelation buffer. Nitrate concentration is a slowly varying,
# persistent signal: its information bleeds months past the edge of a record
# (soil-N legacy, that year's surplus still leaching, multi-month hydrology).
# Two records whose windows merely come within this horizon of each other are
# therefore not statistically independent, so we treat them as overlapping. This
# is the same idea as the embargo/purge band in time-series cross-validation.
_DEFAULT_BUFFER = "100D"


def nitrate_span(site_uid: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (first, last) timestamp of a site's non-null nitrate record.

    Returns None if the site has no nitrate_con column or no observations.
    """
    try:
        water = get_water(site_uid)
    except FileNotFoundError:
        return None
    if "nitrate_con" not in water.columns:
        return None
    series = water["nitrate_con"].dropna()
    if series.empty:
        return None
    return series.index.min(), series.index.max()


def overlaps(span_a, span_b, buffer: str = _DEFAULT_BUFFER) -> bool:
    """Do two nitrate records overlap in time, within an autocorrelation buffer?

    span_a, span_b are (start, end) tuples (e.g. from nitrate_span); either may
    be None, in which case there is no temporal relationship. Two intervals are
    considered to overlap when they intersect *or* fall within ``buffer`` of each
    other — i.e. the calendar gap between them is at most ``buffer``. With
    buffer="0D" this is the plain interval-intersection test.
    """
    if span_a is None or span_b is None:
        return False
    buf = pd.Timedelta(buffer)
    (start_a, end_a), (start_b, end_b) = span_a, span_b
    return (start_a - buf <= end_b) and (start_b - buf <= end_a)


def build_conflict_graph(buffer: str = _DEFAULT_BUFFER) -> nx.Graph:
    """Undirected graph whose edges denote basin containment.

    Nodes <-> sites. An edge a—b is added when one site is contained in the other AND the sites overlap temporally (see the `overlaps` method). Due to some sites being incredibly close, some basins DO actually coincide, creating cycles in this graph.
    """
    spatial = get_basin_graph().to_undirected()
    spans = {s: nitrate_span(s) for s in get_site_ids()}

    conflict = nx.Graph()
    conflict.add_nodes_from(spatial.nodes)
    for a, b in spatial.edges:
        if overlaps(spans.get(a), spans.get(b), buffer=buffer):
            conflict.add_edge(a, b)
    return conflict


def split_groups(buffer: str = _DEFAULT_BUFFER) -> dict[str, int]:
    """Map each site_uid to an integer group id (its conflict component).

    All sites within a group must stay on the same side of any train/test split.
    Sites with no hard conflict are singleton groups.
    """
    conflict = build_conflict_graph(buffer=buffer)
    groups: dict[str, int] = {}
    for gid, component in enumerate(nx.connected_components(conflict)):
        for site in component:
            groups[site] = gid
    return groups


def make_folds(n_splits: int = 5, buffer: str = _DEFAULT_BUFFER, shuffle=False, random_state=None) -> pd.DataFrame:
    """Assign sites to ``n_splits`` folds without breaking any conflict group.

    Uses GroupKFold with the conflict-component id as the group key, so no fold
    boundary ever separates two both-axes-related basins. Returns a DataFrame
    indexed by site_uid with columns ``group`` (conflict component) and ``fold``
    (0..n_splits-1). Fold f is the test set for split f; the rest is train.

    shuffle/random_state are forwarded to GroupKFold. Shuffling only randomizes
    which fold a whole group lands in — groups stay intact, so the no-leak
    guarantee is unaffected; without it the folds are deterministic. (Shuffling
    does not fix the imbalance from the one indivisible mega-group.)
    """
    groups = split_groups(buffer=buffer)
    sites = sorted(groups)
    gids = [groups[s] for s in sites]

    fold = pd.Series(-1, index=sites, dtype=int)
    splitter = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    # GroupKFold needs an X of the right length; the values are unused.
    for f, (_, test_idx) in enumerate(splitter.split(sites, groups=gids)):
        for i in test_idx:
            fold.iloc[i] = f

    out = pd.DataFrame({"group": [groups[s] for s in sites], "fold": fold.values}, index=sites)
    out.index.name = "site_uid"
    return out


def holdout_split(test_size: float = 0.2, buffer: str = _DEFAULT_BUFFER, seed: int = 0):
    """Single train/test split that keeps whole conflict groups together.

    Shuffles the conflict groups (deterministically, via ``seed``) and adds whole
    groups to the test set until it reaches ~``test_size`` of the sites, skipping
    a group that would overshoot the target by more than half a group's slack so
    smaller groups can fill in. Returns (train_sites, test_sites) as sorted lists.

    Note the realized test fraction is quantized by group sizes: if one conflict
    group is larger than ``test_size`` of the data, a clean holdout at that size
    is impossible and the result will be coarser. Prefer make_folds for balanced
    cross-validation; use this for a quick single split.
    """
    import random

    groups: dict[int, list[str]] = {}
    for site, gid in split_groups(buffer=buffer).items():
        groups.setdefault(gid, []).append(site)

    all_sites = sorted(s for members in groups.values() for s in members)
    target = test_size * len(all_sites)

    order = list(groups.values())
    random.Random(seed).shuffle(order)

    test: set[str] = set()
    for members in order:
        if len(test) >= target:
            break
        if not test or len(test) + len(members) <= 1.5 * target:
            test.update(members)
    train = [s for s in all_sites if s not in test]
    return train, sorted(test)


def audit_split(train_sites, test_sites, buffer: str = _DEFAULT_BUFFER) -> pd.DataFrame:
    """List every cross-boundary leak between a train and a test site.

    Returns one row per train/test pair that is spatially related, temporally
    overlapping, or both, with a ``severity`` of 'hard' (both axes — should never
    appear if the split respects conflict groups), 'spatial' or 'temporal'. An
    empty 'hard' subset is the correctness check for make_folds / holdout_split.
    """
    spatial = get_basin_graph().to_undirected()
    spans = {s: nitrate_span(s) for s in set(train_sites) | set(test_sites)}
    train_set, test_set = set(train_sites), set(test_sites)

    rows = []
    for a in train_set:
        for b in spatial.neighbors(a):
            if b not in test_set:
                continue
            spatial_rel = True
            temporal_rel = overlaps(spans.get(a), spans.get(b), buffer=buffer)
            severity = "hard" if temporal_rel else "spatial"
            rows.append({"train_site": a, "test_site": b, "severity": severity})
    return pd.DataFrame(rows, columns=["train_site", "test_site", "severity"])
