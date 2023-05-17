from collections import defaultdict, deque
from copy import deepcopy
from ctypes import Structure, c_float, c_int64
from enum import IntEnum
from heapq import heappop, heappush
from math import inf
from pprint import pprint
from random import choice
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from more_itertools import first

from union_find import UnionFind

# typing aliases
Node = float | str
Edge = Tuple[Node, Node]
Edges = List[Edge]
Path = List[Node]
Predecessors = dict[Node, Optional[Node]]
Distances = dict[Node, float]


class EdgeInfo(Structure):
    _fields_ = [("cap", c_int64), ("flow", c_int64), ("weight", c_float)]

    def __repr__(self):
        return "(C:%.2d F:%.2d)" % (self.cap, self.flow)


class ReturnTypeOption(IntEnum):
    PATH = 1
    PREDECESSORS = 2
    DISTANCES = 3
    ALL = 4
    EXISTS = 5
    PATH_STR = 6


ReturnType = Union[
    Path, Predecessors, Distances, tuple[Distances, Predecessors], bool, list[str]
]


def _gen_path(source: Node, sink: Node, pred: Predecessors) -> Path:
    path = []
    node: Optional[Node] = sink
    while node is not None and node != source:
        path.append(node)
        node = pred[node]
    path.reverse()
    return path


def _gen_path_str(distances: Distances, source: Node, pred: Predecessors) -> list[str]:
    paths = []
    for node in pred:
        if node != source:
            curr = node
            temp: list[str] = []
            while curr is not None and curr != source:
                temp.append(str(curr))
                curr = pred[curr]
            temp.append(str(source))
            temp.reverse()
            if curr is None:
                paths.append("No path from " + str(source) + " to " + str(node))
            else:
                paths.append(
                    "the path from "
                    + str(source)
                    + " to "
                    + str(node)
                    + " is: "
                    + "->".join(temp)
                    + " and the weight is "
                    + " : "
                    + str(distances[node])
                )
    return paths


def _resolve_return_type(
    return_type_option: ReturnTypeOption,
    distances: Distances,
    pred: Predecessors,
    source: Node,
    sink: Optional[Node],
) -> ReturnType:
    match return_type_option:
        case ReturnTypeOption.PATH_STR:
            return _gen_path_str(distances, source, pred)
        case ReturnTypeOption.PREDECESSORS:
            return pred
        case ReturnTypeOption.DISTANCES:
            return distances
        case ReturnTypeOption.ALL:
            return distances, pred
        case ReturnTypeOption.PATH:
            if sink is None:
                raise ValueError("sink must be specified when return type is PATH")
            return _gen_path(source, sink, pred)
        case ReturnTypeOption.EXISTS:
            if sink is None:
                raise ValueError("sink must be specified when return type is NONE")
            path = _gen_path(source, sink, pred)
            return path and first(path) == source
        case _:
            raise ValueError("invalid return type")


class DiGraph(defaultdict):
    """Class representing a digraph"""

    supersource = "S"
    INF = 1 << 64
    __slots__ = ()

    def __init__(self):
        super().__init__(dict)

    def insert_edge(self, edge) -> None:
        src, dst, *rest = edge
        if dst not in self:
            self[dst] = {}
        self[src][dst] = EdgeInfo(*rest)

    def insert_edges_from_iterable(self, edges: Iterable):
        for edge in edges:
            self.insert_edge(edge)

    def set_weights(self, edges, weights):
        if len(edges) != len(weights):
            raise ValueError("number of edges must equal number of weights")
        for edge, weight in zip(edges, weights):
            src, dst = edge
            self[src][dst].weight = weight

    @property
    def vertices(self):
        return self.keys()

    def weight(self, src, dst):
        """Assumes that the first argument is the weight:"""
        return self[src][dst].weight if self.has_edge(src, dst) else self.INF

    def assert_edge_in_graph(self, u, v):
        assert self.has_edge(u, v), f"edge ({u}, {v}) not in graph"

    def random_node(self):
        return choice(tuple(self.keys()))

    def neighbors(self, node):
        if node not in self:
            raise KeyError(f"node {node} not in graph")
        yield from self[node].keys()

    def has_node(self, node):
        return node in self

    def has_edge(self, src, dst):
        if not self.has_node(src):
            return False
        else:
            return dst in self[src]

    def self_loop(self):
        loop_edges = []
        for u in self:
            for v in self[u]:
                if v == u:
                    loop_edges.append(u)
        return loop_edges

    @property
    def edges(self):
        for u in self:
            for v in self[u]:
                yield u, v

    def __repr__(self):
        return repr(self)

    @staticmethod
    def insert_supersource(g):
        if type(g) != dict:
            raise TypeError(
                "this method works with dictionary representations of static graphs"
            )
        g[DiGraph.supersource] = defaultdict(int)

    def bellman_ford(
        self,
        source: Node,
        return_type_option: ReturnTypeOption,
        sink: Optional[Node] = None,
    ) -> ReturnType:
        distances: Distances = defaultdict(lambda: self.INF)
        pred: Predecessors = defaultdict(lambda: None)
        distances[source] = 0

        for i in range(len(self) - 1):  # O(|V| - 1)
            for u in self:  # O(|E|)
                for v in self.neighbors(u):
                    if distances[u] + self.weight(u, v) < distances[v]:  # O(1)
                        distances[v] = self.weight(u, v) + distances[u]
                        pred[v] = u

        for u in self:  # checking for negative-weight edge cycles
            for v in self.neighbors(u):
                if distances[u] + self.weight(u, v) < distances[v]:
                    raise ValueError("the input graph contains a negative-weight cycle")
        return _resolve_return_type(return_type_option, distances, pred, source, sink)

    @property
    def is_empty(self):
        return len(self) == 0

    def dijkstra(
        self,
        source: Node,
        return_type_option: ReturnTypeOption,
        w: Optional[Callable[[Node, Node], float]] = None,
        target: Optional[Node] = None,
    ):
        """Neat implementation of dijkstra"""
        w = self.weight if w is None else w  # use the default weight function
        distances: Distances = defaultdict(lambda: self.INF)
        pred: Predecessors = defaultdict(lambda: None)
        distances[source] = 0
        Q: list[tuple[float, Node]] = [(0, source)]

        while Q:
            _, u = heappop(Q)  # extract_min
            if u == target:
                break
            for v in self.neighbors(u):
                if distances[u] + w(u, v) < distances[v]:  # relaxation
                    distances[v] = distances[u] + w(u, v)
                    pred[v] = u
                    heappush(Q, (distances[v], v))

        return _resolve_return_type(return_type_option, distances, pred, source, target)

    def pop_supersource(self, D):
        for u in D:
            D[u].pop(self.supersource)
        D.pop(self.supersource)

    def floyd_warshall(self, path=False):
        if path:
            return self._fw_with_path()
        else:
            return self._fw_without_path()

    def _fw_without_path(self):
        """All pairs shortest path algorithm
        Better for dense graphs"""
        N = len(self)
        d = np.empty((N, N))

        for i in self.vertices:
            for j in self.vertices:
                d[i][j] = self.weight(i, j)

        for i in self.vertices:
            d[i][i] = 0

        for k in self.vertices:
            for j in self.vertices:
                for i in self.vertices:
                    x = d[i][j]
                    y = d[i][k] + d[k][j]
                    if y < x:
                        d[i][j] = y

    def _fw_with_path(self):
        """All pairs shortest path algorithm
        Better for dense graphs"""
        N = len(self)
        d = np.empty((N, N))
        nxt = defaultdict(dict)

        for i in self.vertices:
            for j in self.vertices:
                d[i][j] = self.weight(i, j)
                if d[i][j] != self.INF:
                    nxt[i][j] = j
                else:
                    nxt[i][j] = None

        for i in self.vertices:
            d[i][i] = 0

        for k in self.vertices:
            for j in self.vertices:
                for i in self.vertices:
                    x = d[i][j]
                    y = d[i][k] + d[k][j]
                    if y < x:
                        d[i][j] = y
                        nxt[i][j] = nxt[i][k]
        return d, nxt

    @staticmethod
    def reconstruct(u, v, nxt):
        if nxt[u][v] == -1:
            return []
        path = [u]
        while u != v:
            u = nxt[u][v]
            path.append(u)
        return path

    def a_star(
        self,
        source: Node,
        target: Node,
        return_type: ReturnTypeOption,
        h: Callable[[Node], float] = lambda x: 0,
    ) -> ReturnType:
        """A* is guided by a heuristic function h(n) which is an estimate of the distance from the
        current node n to the goal node
         g(n) is a heuristic function specifying the length of the shortest path from the source to node n

        As an example, when searching for the shortest route on a map,
        h(x) might represent the straight-line distance to the goal,
        since that is physically the smallest possible distance between any two points.

        If the heuristic h satisfies the additional condition h(x) ≤ d(x, y) + h(y) for every edge (x, y)
        of the graph (where d denotes the length of that edge), then h is called monotone, or consistent.
        With a consistent heuristic,
        A* is guaranteed to find an optimal path without processing any node more than once
        and A* is equivalent to running Dijkstra's algorithm with the reduced cost d'(x, y) = d(x, y) + h(y) − h(x).

        Dijkstra can be viewed as a special case of A* where h(n) = 0, because it is a greedy algorithm. It makes the best
        choice locally with no regards to the future"""

        assert not self.is_empty, "Empty graph."
        assert self.has_node(source), f"Missing source {source}."
        assert self.has_node(target), f"Missing target {target}."

        pred: Predecessors = defaultdict(lambda: None)

        g_score: Distances = defaultdict(lambda: self.INF)
        g_score[source] = 0

        f_score: Distances = defaultdict(lambda: self.INF)
        f_score[source] = h(source)

        Q: list[tuple[float, Node]] = [(f_score[source], source)]
        fringe = {source}  # openSet

        while Q:
            _, u = heappop(Q)
            if u == target:
                return _resolve_return_type(return_type, g_score, pred, source, target)
            fringe.remove(u)
            for v in self.neighbors(u):
                temp_g = g_score[u] + self.weight(u, v)
                if temp_g < g_score[v]:
                    pred[v] = u
                    g_score[v] = temp_g
                    f_score[v] = g_score[v] + h(v)
                    if v not in fringe:
                        fringe.add(v)
                        heappush(Q, (f_score[v], v))

        raise ValueError("Goal was never reached.")

    def johnsons(self):
        """All-pairs shortest paths"""
        g = deepcopy(self)  # we dont want to modify the original graph
        g.insert_supersource(g)

        path_exists, h = self.bellman_ford(
            DiGraph.supersource, return_type_option=ReturnTypeOption.EXISTS
        )
        if not path_exists:
            return False
        else:
            w_hat = defaultdict(dict)
            for u, v in g.edges:
                w_hat[u][v] = g.weight(u, v) + h[u] - h[v]

            D = defaultdict(dict)
            for u in g:
                du_prime = g.dijkstra(
                    u,
                    w=lambda x, y: w_hat[x][y],
                    return_type_option=ReturnTypeOption.DISTANCES,
                )
                for v in g:
                    D[u][v] = du_prime[v] + h[v] - h[u]
            self.pop_supersource(D)
            return dict(D)

    def bfs(self, source):
        """Queue based bfs. Returns a predecessor dictionary."""

        discovered = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        discovered[source] = 0

        q = deque([source])
        while q:
            u = q.pop()
            for v in self.neighbors(u):
                if discovered[v] == self.INF:
                    discovered[v] = discovered[u] + 1
                    pred[v] = u
                    q.appendleft(v)
        return pred

    def dls(self, u, depth, target=None):
        """perform depth-limited search"""
        pass

    def iddfs(self, source, max_depth):
        """Iterative deepening depth first search"""
        pass

    def euler_tour(self, source, print_path=False):
        """Works for DAGS"""
        visited = set()
        order = []
        seen = set()

        def euler_visit(u, order_):
            order_.append(u)
            seen.add(u)
            for v in self.neighbors(u):
                if v not in visited and v not in seen:
                    euler_visit(v, order_)
            visited.add(u)
            if len(list(self.neighbors(u))) != 0:
                order_.append(u)

        euler_visit(source, order)
        if print_path:
            pprint("->".join(order))
        return order

    def reversed(self):
        rev_g = self.__new__(self.__class__)
        for edge in self.edges:
            src, dst, *rest = edge
            rev_g.insert_edge((dst, src, *rest))
        return rev_g

    def korasaju_scc(self):
        times, _ = self.iterative_dfs(self.vertices, self)
        _, components = self.iterative_dfs(reversed(times), self.reversed())
        return components

    def topsort(self):
        times, _ = self.iterative_dfs(self.vertices, self)
        return times

    @staticmethod
    def iterative_dfs(V, g):
        """times specifies the order in which the dfs was finished"""
        finished, sccs, explored = [], [], set()

        for w in V:
            scc = []
            S = [(False, w)]

            while S:
                completed, u = S.pop()
                # check if already processed
                if completed:
                    finished.append(u)
                    continue

                elif u in explored:
                    continue

                # mark the node
                scc.append(u)
                explored.add(u)

                # search in depth
                S.append((True, u))  # this node has finished being traversed
                S.extend((False, v) for v in g.neighbors(u))

            if scc:
                sccs.append(scc)

        return finished, sccs

    def get_in_degree(self):
        """calculate the indegrees"""
        in_deg = defaultdict(int)
        for src in self:
            for dst in self[src]:
                in_deg[dst] += 1
        return in_deg

    def kahn(self):
        # first find all the vertices with no incoming edges
        in_deg = self.get_in_degree()
        queue = deque([v for v in self.vertices if not in_deg[v]])
        sorted_list = []
        nvisited = 0
        while queue:
            u = queue.pop()
            sorted_list.append(u)
            nvisited += 1
            for v in self[u]:
                in_deg[v] -= 1
                if not in_deg[v]:
                    queue.appendleft(v)
        if nvisited != len(self):
            raise ValueError("the graph is cyclic")
        return sorted_list

    def recursive_dfs(self, source, sort=True):
        """Recursive dfs"""

        visited = set()
        d = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        d[source] = 0

        def visit(cur_node, time, top):
            time += 1
            d[cur_node] = time
            for v in self.neighbors(cur_node):
                if v not in visited and d[v] == inf:
                    pred[v] = cur_node
                    visit(v, time, top)
            visited.add(cur_node)

            if top:
                top.appendleft(cur_node)

        times = None if not sort else deque()
        for u in self:
            if u not in visited:
                visit(u, sort, times)
        return times, pred


class Graph(DiGraph):
    def __init__(self):
        super().__init__()

    def insert_edge(self, edge: Tuple) -> None:
        src, dst, *rest = edge
        edge_info = EdgeInfo(*rest)
        self[src][dst] = edge_info
        self[dst][src] = edge_info

    def prim(self) -> Edges:
        """Find a minimum spanning tree of a graph g using Prim's algorithm.
        v.key = min {w(u,v) | u in S}

        Running Time: O(E lg V) using a binary heap."""

        V = set(self)
        s = V.pop()
        Q = [(0, s)]

        key = defaultdict(lambda: self.INF)
        key[s] = 0
        pred: Predecessors = defaultdict(lambda: None)

        S = set()

        while Q:  # |Q| = |V|
            _, u = heappop(Q)  # O (lg |V|)
            S.add(u)
            for v in self.neighbors(u):  # O(deg[v]) ... ∑ (deg(v)) ∀ v ⊆ V = O(|E|)
                if v not in S and self.weight(u, v) < key[v]:  # O(1)
                    key[v] = self.weight(u, v)
                    heappush(Q, (key[v], v))  # O(lg |V|)
                    pred[v] = u

        mst = [(v, pred[v]) for v in V if pred[v]]
        return mst

    def kruskal(self) -> Edges:
        """Kruskal's algorithm."""

        V = self
        T = UnionFind([v for v in V])  # O(|V|) make-set() calls
        # O(|E| lg |E|) or O(|E|) when using counting sort if weights are integer weights in the range O(|E|^O(1))
        E = sorted(self.edges, key=lambda x: self.weight(*x))
        mst = []

        for u, v in E:  # O(|E|)
            if T[u] != T[v]:  # amortized O(⍺(V))
                mst.append((u, v))
                T.union(u, v)
        return mst

    def boruvka(self):
        # TODO
        pass
