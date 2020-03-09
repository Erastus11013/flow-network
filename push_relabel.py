from typing import Dict, Tuple, Union, Iterable
from collections import defaultdict, deque
from functools import reduce
from ctypes import Structure, c_int, c_float
from numpy import random
from pprint import pprint
from random import choice
from heapq import heappush, heappop
from itertools import chain
from operator import add
from copy import deepcopy

Node = object


class EdgeInfo(Structure):
    _fields_ = [("cap", c_int), ("flow", c_int), ("weight", c_float)]

    def __repr__(self):
        return "EdgeInfo (C:%.2d F:%.2d)" % (self.cap, self.flow)


Graph = Dict[Node, Dict[Node, EdgeInfo]]
ResidualGraph = Dict[Node, Dict[Node, float]]

excess = defaultdict(lambda: 0)
height = None


def edge_in_graph(src: Node, dst: Node, g: Graph):
    if src not in g:
        return False
    return dst in g[src]


def edges(g: Graph):
    a = [tuple((n, neighbor) for neighbor in adjacency(g, n)) for n in g]
    return chain(*a)


def insert_edges_from_iterable(g: Graph, edges: Iterable[tuple]):
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


def dijkstra(g, source, target=None, weight_function=None) -> Dict[Node, float]:
    pred, distance = defaultdict(lambda: None), defaultdict(lambda: int('inf'))
    distance[source] = 0
    W = weight if weight_function is None else weight_function

    Q = [(0, source)]
    while Q:
        u = heappop(Q)[1]
        if u == target:
            break
        for v in adjacency(g, u):
            if distance[u] > distance[v] + W(u, v, g):
                distance[u] = distance[v] + W(u, v, g)
                heappush((distance[v], v))
                pred[v] = u

    return distance


def residual_graph(g: Graph) -> ResidualGraph:
    rg = defaultdict(dict)
    for u, v in edges(g):
        if (rc := residual_capacity(g, u, v)) != 0:
            rg[u][v] = rc
        rg[v][u] = flow(g, u, v)
    return rg


def rand_flow_cap(lim) -> tuple:
    cap = random.randint(2, lim)
    return cap, random.randint(1, cap)


def node_is_active(u, source, sink) ->bool:
    return u != source != sink and excess[u] > 0


def bfs(graph: Union[ResidualGraph, Graph], s: Node) -> Dict[Node, float]:
    distance = defaultdict(lambda: 0)
    distance[s] = 0
    Q = deque([s])
    while Q:
        u = Q.pop()
        for v in adjacency(graph, u):
            if not distance[v]:
                distance[v] = distance[u] + 1
    return distance


def shallow_reverse(g) -> Dict[Node, Dict[Node, int]]:
    revg = defaultdict(dict)
    for u in g:
        for v in adjacency(g, u):
            revg[v][u] = 0
    return revg


def exists_in_residual_graph(g: Graph, u: Node, v: Node) -> bool:
    if edge_in_graph(u, v, g):
        return g[u][v].cap - g[u][v].flow > 0
    return False


def get_active_node():
    pass


def generic_push_relabel(graph: Graph, source: Node, sink: Node) -> float:
    """
    Invariants:
     1. h(s) = n at all times (where n = |V|);
     2. h(t) = 0;
     3. for every edge (v, w) of the current residual network (with positive residual capacity), h(v) ≤ h(w) + 1.

     excess(u) = flow(into) - flow(out)
    """

    assert graph is not None and source in graph

    g = deepcopy(graph)
    global height
    # saturate all the edges coming out of the source
    for v in adjacency(g, source):
        excess[v] = g[source][v].flow = g[source][v].cap
        excess[source] -= g[source][v].cap

    # initialize heights
    for u, v in edges(g):
        # insert reverse edges
        if g[u][v].cap - g[u][v].flow > 0:
            insert_edge(g, (v, u, g[u][v].cap, g[u][v].cap - g[u][v].flow))
    height = bfs(shallow_reverse(g), sink)
    height[source] = num_nodes(g)

    # create the the residual graph of g, Q contains the nodes with positive excesses
    # Initially, those are just the edges outgoing from the source
    Q = [(-height[u], u) for u in adjacency(g, source)]
    while Q:
        u = heappop(Q)[1]
        if excess[u] and u != sink and u != source:
            if not _push(g, u, Q):
                _relabel(g, u, Q)

    return sum(g[source][v].flow for v in adjacency(g, source))


def _push(g: Graph, u: Node, queue) -> bool:
    """ Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
        from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
            by δ. If δ = cf (uv), this is called a saturating push, else it is an non-saturating push.
    """

    pushed = False
    for v in adjacency(g, u):
        if excess[u] > 0 and height[u] == height[v] + 1:
            pushed = True
            delta = min(g[u][v].cap - g[u][v].flow, excess[u])
            if edge_in_graph(u, v, g):
                g[u][v].flow += delta
            else:
                g[v][u].flow -= delta
            excess[u] -= delta
            excess[v] += delta
            # excess[v] must be > 0
            if v != u:
                heappush(queue, (-height[v], v))
        if excess[u] == 0:
            return True
    return pushed


def _relabel(g: Graph, u: Node, queue):
    valid = [v for v in adjacency(g, u) if g[u][v].cap - g[u][v].flow > 0]
    height[u] = min(height[v] for v in valid) + 1
    heappush(queue, (-height[u], u))


def test(nn):
    g = defaultdict(dict)
    rand_nodes = random.randint(0, 100, nn)
    e = list((choice(rand_nodes), choice(rand_nodes),
              *rand_flow_cap(100)) for _ in range(nn << 3))
    insert_edges_from_iterable(g, e)
    pprint(g)
    pprint(residual_graph(g))


def test_generic_push_relabel():
    s, t, a, b, c, d, e, f, g, h, i = 's', 't', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'
    # n = [s, t, a, b, c, d, e, f, g, h, i]
    e = [(s, a, 5), (s, d, 10), (s, g, 15), (a, b, 10), (b, c, 10), (b, e, 25), (c, t, 5), (d, a, 15), (d, e, 20),
         (e, f, 30), (e, g, 5),
         (f, t, 15), (f, b, 15), (f, i, 15), (g, h, 25), (h, i, 10), (h, f, 20), (i, t, 10)]

    s, v1, v2, v3, v4, v5, t = 's', 'v1', 'v2', 'v3', 'v4', 'v5', 't'
    # n1 = [s, v1, v2, v3, v4, t]
    e1 = [(s, v1, 16), (s, v2, 13), (v1, v3, 12), (v2, v1, 4), (v2, v4, 14), (v3, v2, 9), (v3, t, 20),
          (v4, v3, 7), (v4, t, 4)]

    g = defaultdict(dict)
    insert_edges_from_iterable(g, e1)
    maxflow = generic_push_relabel(g, s, t)
    print(maxflow)


if __name__ == "__main__":
    test_generic_push_relabel()
