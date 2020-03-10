from typing import Dict, Tuple, Union, Iterable
from types import FunctionType
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
    _fields_ = [("cap", c_float), ("flow", c_float), ("weight", c_float)]

    def __repr__(self):
        return "(C:%.2d F:%.2d)" % (self.cap, self.flow)


Graph = Dict[Node, Dict[Node, EdgeInfo]]
ResidualGraph = Dict[Node, Dict[Node, float]]

excess = defaultdict(lambda: 0)
height = None
inqueue = defaultdict(lambda: False)
seen = defaultdict(lambda: 0)


def edge_in_graph(src: Node, dst: Node, g: Graph) -> bool:
    if src not in g:
        return False
    return dst in g[src]


def edges(g: Graph) -> Iterable[Tuple[Node, Node]]:
    a = [tuple((n, neighbor) for neighbor in adjacency(g, n)) for n in g]
    return chain(*a)


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


def dijkstra(g: Union[Graph, ResidualGraph], source: Node,
             target: Node = None, weight_function: FunctionType = None) -> Dict[Node, float]:
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


def rand_flow_cap(lim: int) -> tuple:
    cap = random.randint(2, lim)
    return cap, random.randint(1, cap)


def node_is_active(u: Node, source: Node, sink: Node) -> bool:
    return u != source and u != sink and excess[u] > 0


def bfs(graph: Union[ResidualGraph, Graph], s: Node) -> Dict[Node, float]:
    distance = defaultdict(lambda: 0)
    distance[s] = 0
    Q = deque([s])
    while Q:
        u = Q.pop()
        for v in adjacency(graph, u):
            if not distance[v]:
                distance[v] = distance[u] + 1
                Q.appendleft(v)
    return distance


def shallow_reverse(g: Union[Graph, ResidualGraph]) -> Dict[Node, Dict[Node, int]]:
    revg = defaultdict(dict)
    for u, v in edges(g):
        revg[v][u] = 0
    return revg


def initialize_preflow(graph: Graph, source: Node, sink: Node) -> Graph:
    assert graph is not None and source in graph
    g = deepcopy(graph)
    global height

    for u, v in edges(g):
        g[u][v].flow = 0

    # saturate all the edges coming out of the source
    for v in adjacency(g, source):
        excess[v] = g[source][v].flow = g[source][v].cap
        excess[source] -= g[source][v].cap

    for u, v in edges(g):
        insert_edge(g, (v, u, g[u][v].cap, residual_capacity(g, u, v), 1))

    # Initialize heights as the shortest distance from the sink to every node except the source
    # We perform bfs on the original graph, not the residual one
    height = bfs(shallow_reverse(graph), sink)
    height[source] = num_nodes(g)

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
    Q = []
    for u in adjacency(g, source):
        inqueue[u] = True
        heappush(Q, (-height[u], u))

    while Q:
        # highest active node, has the lowest -(height)
        u = heappop(Q)[1]
        inqueue[u] = False
        for v in adjacency(g, u):
            if excess[u] == 0:
                break
            if height[u] == height[v] + 1 and residual_capacity(g, u, v) > 0:
                push(g, u, v)
                if not inqueue[v] and v not in (source, sink):
                    heappush(Q, (-height[v], v))
                    inqueue[v] = True
        if excess[u] > 0:
            relabel(g, u)
            heappush(Q, (-height[u], u))
            inqueue[u] = True

    return sum(g[source][v].flow for v in adjacency(g, source))


def relabel_to_front(graph: Graph, source: Node, sink: Node) -> float:
    """Relabel to front algorithm"""

    # list of valid nodes
    L = list(filter(lambda node: node not in (sink, source), graph.keys()))
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
    """ Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
        from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
            by δ. If δ = cf (uv), this is called a saturating push, else it is an non-saturating push.
    """
    assert excess[u] > 0 and height[u] == height[v] + 1
    delta = min(residual_capacity(g, u, v), excess[u])
    g[u][v].flow += delta
    g[v][u].flow -= delta
    excess[u] -= delta
    excess[v] += delta


def relabel(g: Graph, u: Node):
    valid = [v for v in adjacency(g, u) if residual_capacity(g, u, v) > 0]
    assert (len(valid) != 0)
    assert excess[u] > 0 and all(height[u] <= height[v] for v in valid)
    height[u] = min(height[v] for v in valid) + 1


def discharge(g: Graph, u: Node):
    neighbors = adjacency(g, u)
    while excess[u] > 0:
        if seen[u] < len(neighbors):
            v = neighbors[seen[u]]
            if residual_capacity(g, u, v) > 0 and height[u] == height[v] + 1:
                push(g, u, v)
            else:
                seen[u] += 1
        else:
            relabel(g, u)
            seen[u] = 0


def test(nn: int):
    g = defaultdict(dict)
    rand_nodes = random.randint(0, 100, nn)
    e = list((choice(rand_nodes), choice(rand_nodes),
              *rand_flow_cap(100)) for _ in range(nn << 3))
    insert_edges_from_iterable(g, e)
    pprint(g)
    pprint(residual_graph(g))


# s, t, a, b, c, d, e, f, g, h, i = 's', 't', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'
# # n = [s, t, a, b, c, d, e, f, g, h, i]
# e = [(s, a, 5), (s, d, 10), (s, g, 15), (a, b, 10), (b, c, 10), (b, e, 25), (c, t, 5), (d, a, 15), (d, e, 20),
#         (e, f, 30), (e, g, 5),
#         (f, t, 15), (f, b, 15), (f, i, 15), (g, h, 25), (h, i, 10), (h, f, 20), (i, t, 10)]

s, v1, v2, v3, v4, v5, t = 's', 'v1', 'v2', 'v3', 'v4', 'v5', 't'
# n1 = [s, v1, v2, v3, v4, t]
e = [(s, v1, 16), (s, v2, 13), (v1, v3, 12), (v2, v1, 4), (v2, v4, 14), (v3, v2, 9), (v3, t, 20),
     (v4, v3, 7), (v4, t, 4)]


def test_generic_push_relabel():
    g = defaultdict(dict)
    insert_edges_from_iterable(g, e)
    mf = fifo_push_relabel(g, s, t)
    # print(mf)


def test_relabel_to_front():
    g = defaultdict(dict)
    insert_edges_from_iterable(g, e)
    mf = relabel_to_front(g, s, t)
    # print(mf)


if __name__ == "__main__":
    from time import perf_counter
    num_iter = 100
    t1 = perf_counter()
    for _ in range(num_iter):
        excess = defaultdict(lambda: 0)
        inqueue = defaultdict(lambda: False)
        height = None
        test_generic_push_relabel()
    print(f"FIFO push relabel ran in: {perf_counter() - t1:.2f} seconds.")

    t1 = perf_counter()
    for _ in range(num_iter):
        excess = defaultdict(lambda: 0)
        height = None
        seen = defaultdict(lambda: 0)
        test_relabel_to_front()
    print(f"Relabel to front ran in: {perf_counter() - t1:.2f} seconds.")
