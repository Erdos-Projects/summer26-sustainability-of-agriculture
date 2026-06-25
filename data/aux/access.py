"""Read-only access layer for auxiliary derived data in data/aux/."""

import sys
from pathlib import Path

import pandas as pd
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_GRAPH_FILE = _THIS_DIR / "aux_data" / "basin_containment_graph.parquet"

sys.path.insert(0, str(_THIS_DIR.parents[1]))
from data.water import get_site_ids


def get_basin_graph(immediate_only: bool = False) -> nx.DiGraph:
    """Load the basin-containment graph as a directed graph.

    Each edge ``child -> parent`` means "child's monitoring sensor lies inside
    parent's basin", i.e. child is nested in parent. Built and persisted by
    make_aux.save_basin_graph as an edge list in aux_data/basin_graph.parquet.

    Every site (including roots with no enclosing basin) is present as a node, so
    the graph always has one node per site; only the edges come from the parquet.
    Each parent edge keeps the parent basin's area as the edge attribute
    ``parent_area`` (m^2).

    Parameters
    ----------
    immediate_only : bool, default False
        If True, keep for each child only the edge to its smallest enclosing
        basin (minimum parent_area), turning the full containment relation into a
        forest with at most one parent per node — the immediate-parent tree to
        split train/test/holdout on. If False, return the full relation (a child
        points to every basin that encloses it).

    Returns
    -------
    networkx.DiGraph
        Nodes are site_uids; edges are child -> parent with a parent_area attr.

    Notes
    -----
    Useful views once loaded:
      * ``nx.descendants(g, s)``           -> every basin s is nested within
      * ``[n for n in g if g.out_degree(n) == 0]`` -> root (maximal) basins
      * ``nx.weakly_connected_components(g)`` -> basin families (holdout units)
      * ``nx.topological_generations(g)``   -> nesting depth layers

    Raises FileNotFoundError if the graph has not been generated yet
    (run make_aux.save_basin_graph / make_aux.main).
    """
    edges = pd.read_parquet(_GRAPH_FILE)

    if immediate_only and not edges.empty:
        edges = edges.loc[edges.groupby("child")["parent_area"].idxmin()]

    g = nx.DiGraph()
    g.add_nodes_from(get_site_ids())
    for child, parent, parent_area in edges.itertuples(index=False):
        g.add_edge(child, parent, parent_area=parent_area)
    return g
