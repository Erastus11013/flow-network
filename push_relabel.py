from collections import defaultdict, deque
from functools import reduce
from heapq import heappop, heappush
from itertools import chain
from operator import add
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from core import Distances, EdgeInfo, Node

Graph = Dict[Node, Dict[Node, EdgeInfo]]
ResidualGraph = Dict[Node, Dict[Node, float]]

excess: dict[Node, int] = defaultdict(int)
height: Dict[Node, float] = {}
queued: set[Node] = set()
seen: dict[Node, int] = defaultdict(int)


def edge_in_graph(src: Node, dst: Node, g: Graph) -> bool:
    if src not in g:
        return False
    return dst in g[src]


def edges(g: Graph) -> Iterable[Tuple[Node, Node]]:
    a = [tuple((n, neighbor) for neighbor in adjacency(g, n)) for n in g]
    return chain(*a)


def nodes(g: Graph):
    n = set()
    for u in g:
        for v in g[u]:
            n.add(v)
    return n


def insert_edges_from_iterable(g: Graph, edges: Iterable[tuple]) -> None:
    """Assumes that the nodes are in the order.
    (src, dst, attributes)
    """
    for edge in edges:
        insert_edge(g, edge)


def num_nodes(g: Graph) -> int:
    return len(g)


def adjacency(g: Union[Graph, ResidualGraph], n: Node) -> Tuple:
    if n in g:
        return tuple(g[n].keys())
    return ()


def num_edges(g: Graph) -> int:
    return reduce(add, map(lambda n: len(g[n]), g.keys()))


def insert_edge(g: Graph, edge) -> None:
    src, dst, *rest = edge
    g[src][dst] = EdgeInfo(*rest)


def assert_edge_in_graph(u, v, g) -> None:
    if not edge_in_graph(u, v, g):
        raise ValueError(f"Edge ({repr(u)}, {repr(v)}) not in graph")


def weight(u, v, g) -> int:
    assert_edge_in_graph(u, v, g)
    return g[u][v].weight


def capacity(g, u, v) -> int:
    assert_edge_in_graph(u, v, g)
    return g[u][v].cap


def flow(g, u, v) -> int:
    assert_edge_in_graph(u, v, g)
    return g[u][v].flow


def residual_capacity(g, u, v) -> int:
    return capacity(g, u, v) - flow(g, u, v)


def residual_graph(g: Graph) -> ResidualGraph:
    rg: ResidualGraph = defaultdict(dict)
    for u, v in edges(g):
        if (rc := (g[u][v].cap - g[u][v].flow)) != 0:
            rg[u][v] = rc
        rg[v][u] = flow(g, u, v)
    return rg


def rand_flow_cap(lim: int) -> tuple:
    cap = np.random.randint(2, lim)
    return cap, np.random.randint(1, cap)


def node_is_active(u: Node, source: Node, sink: Node) -> bool:
    return u != source and u != sink and excess[u] > 0


def bfs(graph: Union[ResidualGraph, Graph], s: Node) -> Dict[Node, float]:
    distance: Distances = defaultdict(int)
    Q = deque([s])
    while Q:
        u = Q.pop()
        for v in adjacency(graph, u):
            if not distance[v]:
                distance[v] = distance[u] + 1
                Q.appendleft(v)
    return distance


def shallow_reverse(g: Union[Graph, ResidualGraph]) -> Dict[Node, Dict[Node, int]]:
    revg: Dict[Node, Dict[Node, int]] = defaultdict(dict)
    for u, v in edges(g):
        revg[v][u] = 0
    return revg


def initialize_preflow(g: Graph, source: Node, sink: Node) -> Graph:
    assert g is not None and source in g
    global height

    # Initialize heights as the shortest distance from the sink to every node except the source
    # We perform bfs on the original graph, not the residual one

    height = bfs(shallow_reverse(g), sink)
    height[source] = num_nodes(g)

    for u, v in edges(g):
        g[u][v].flow = 0

    # saturate all the edges coming out of the source
    for v in adjacency(g, source):
        excess[v] = g[source][v].flow = g[source][v].cap
        excess[source] -= g[source][v].cap

    for u, v in edges(g):
        insert_edge(g, (v, u, g[u][v].cap, (g[u][v].cap - g[u][v].flow), 1))

    # Create the the residual graph of g, Q contains the nodes with positive excesses
    # Initially, those are just the edges outgoing from the source
    return g


def fifo_push_relabel(graph: Graph, source: Node, sink: Node) -> Tuple[float, Graph]:
    """
    FIFO Push/Relabel
    Heuristics used:
        1. Choosing highest vertex
        2. Initializing the heights to shortest v-t paths
    Invariants:
        1. h(s) = n at all times (where n = |V|);
        2. h(t) = 0;
        3. for every edge (v, w) of the current residual network (with positive residual capacity), h(v) ≤ h(w) + 1.
    excess(u) = flow(into) - flow(out)
    residual_capacity(u, v) = capacity(u, v) - flow(u, v)

    """
    g = initialize_preflow(graph, source, sink)
    Q: list[tuple[float, Node]] = []
    for u in adjacency(g, source):
        queued.add(u)
        heappush(Q, (-height[u], u))
    while Q:
        # highest active node, has the lowest -(height)
        u = heappop(Q)[1]
        queued.discard(u)
        if u == sink:
            continue
        for v in adjacency(g, u):
            if excess[u] == 0:
                break
            if height[u] == height[v] + 1 and (g[u][v].cap - g[u][v].flow) > 0:
                push(g, u, v)
                if v not in queued and v not in (source, sink):
                    heappush(Q, (-height[v], v))
                    queued.add(v)
        if excess[u] > 0:
            relabel(g, u)
            heappush(Q, (-height[u], u))
            queued.add(u)

    return sum(g[source][v].flow for v in adjacency(g, source))


def relabel_to_front(graph: Graph, source: Node, sink: Node) -> float:
    """Relabel to front algorithm"""

    # list of valid nodes
    L = list(filter(lambda node: node not in (source, sink), nodes(graph)))
    g = initialize_preflow(graph, source, sink)
    p, n = 0, len(L)

    while p < n:
        u = L[p]
        old_height = height[u]
        discharge(g, u)
        if height[u] > old_height:
            L.insert(0, L.pop(p))  # move to front
            p = 0
        else:
            p += 1

    return sum(g[source][v].flow for v in adjacency(g, source))


def push(g: Graph, u: Node, v: Node):
    """Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
    from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
        by δ. If δ = cf (uv), this is called a saturating push, else it is an non-saturating push.
    """
    # assert excess[u] > 0 and height[u] == height[v] + 1
    delta = min((g[u][v].cap - g[u][v].flow), excess[u])
    g[u][v].flow += delta
    g[v][u].flow -= delta
    excess[u] -= delta
    excess[v] += delta


def relabel(g: Graph, u: Node):
    valid = [v for v in adjacency(g, u) if (g[u][v].cap - g[u][v].flow) > 0]
    # assert (len(valid) != 0)
    # assert excess[u] > 0 and all(height[u] <= height[v] for v in valid)
    height[u] = min(height[v] for v in valid) + 1


def discharge(g: Graph, u: Node):
    neighbors = adjacency(g, u)
    while excess[u] > 0:
        if seen[u] < len(neighbors):
            v = neighbors[seen[u]]
            if (g[u][v].cap - g[u][v].flow) > 0 and height[u] == height[v] + 1:
                push(g, u, v)
            else:
                seen[u] += 1
        else:
            relabel(g, u)
            seen[u] = 0
