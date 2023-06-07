from typing import Iterable

from core import FlowNetwork, Node, Predecessors, super_sink, super_source
from solvers import get_default_solver


def maximum_bipartite_matching(
    A: list[Node], B: list[Node], E: list[tuple[Node, Node, int]], max_cap: int
) -> list[tuple[Node, Node]]:
    """Takes at input the left and right edges
    Given a bipartite graph G = (A ∪ B, E), find an S ⊆ A × B that is
    a matching and is as large as possible."""

    digraph = FlowNetwork()
    digraph.insert_edges_from_iterable(E)
    if not digraph.is_bipartite():
        raise ValueError("The graph must be bipartite for maximum bipartite matching.")
    # add a super sink and a super source
    for source in A:
        digraph.insert_edge(super_source, source, max_cap)
    for sink in B:
        digraph.insert_edge(sink, super_sink, max_cap)
    solver = get_default_solver()(digraph)
    _ = solver.solve(super_source, super_sink)
    matching = [(u, v) for (u, v, _) in E if solver.graph[u][v].flow == 1]
    return matching


def edge_disjoint_paths(digraph: FlowNetwork, source: Node, sink: Node) -> Iterable:
    """Given directed graph G, and two nodes s and t, find k paths from
        s to t such that no two paths share an edge.

    Menger’s Theorem: Given a directed graph G with nodes s,t the maximum number of
        edge-disjoint s-t paths equals the minimum number of edges whose
        removal separates s from t.

    Suppose you want to send k large files from s to t but never have two files use
        the same network link (to avoid congestion on the links).
    """

    for u in digraph:
        for v in digraph[u]:
            digraph[u][v].cap = 1

    get_default_solver()(digraph).solve(source, sink)

    # use dfs to find the paths
    S, paths = [source], []
    visited: set[Node] = set()
    pred: Predecessors = Predecessors()

    while S:
        u = S.pop()
        if u == sink:
            path = [sink]
            current = pred[sink]
            while current is not None:
                path.append(current)
                current = pred[current]
            paths.append(tuple(reversed(path)))
            continue
        if u in visited:
            continue
        visited.add(u)
        for v in digraph.adjacency(u):
            if u not in visited and digraph[u][v].flow:
                S.append(v)
                pred[v] = u
    return iter(paths)
