from heapq import heappush, heappop
from pprint import pprint
from collections import deque, defaultdict
from math import inf
from copy import deepcopy
from random import randint
from copy import deepcopy
from itertools import count
import numpy as np
from union_find import UnionFind
from typing import Iterable, Tuple, List
from types import FunctionType
from ctypes import Structure, c_int64, c_float

# typing aliases
Node = object
Edge = Tuple[Node, Node]
Edges = List[Edge]
Path = List[Node]


class EdgeInfo(Structure):
    _fields_ = [("cap", c_int64), ("flow", c_int64), ("weight", c_float)]

    def __repr__(self):
        return "(C:%.2d F:%.2d)" % (self.cap, self.flow)


class Graph:
    """Class representing a digraph"""
    supersource = 'S'
    INF = (1 << 10)

    def __init__(self):
        self.nodes = defaultdict(dict)

    def insert_edge(self, edge) -> None:
        src, dst, *rest = edge
        self.nodes[src][dst] = EdgeInfo(*rest)

    def insert_edges_from_iterable(self, edges: Iterable):
        for edge in edges:
            self.insert_edge(edge)

    def set_weights(self, edges, weights):
        if len(edges) != len(weights):
            raise ValueError("number of edges must equal number of weights")
        for edge, weight in zip(edges, weights):
            src, dst = edge
            self.nodes[src][dst].weight = weight

    def weight(self, src, dst):
        """Assumes that the first argument is the weight:"""
        if not self.has_edge(src, dst):
            return self.INF
        y = self.nodes[src][dst]
        if type(y) == tuple:
            return y[0]
        return y

    def assert_edge_in_graph(self, u, v):
        assert self.has_edge(u, v), f"edge ({u}, {v}) not in graph"

    def random_node(self):
        i = 0
        j = randint(0, len(self.nodes.keys()) - 1)
        assert j < len(self.nodes)
        for k in self.nodes.keys():
            if i == j:
                return k
            i += 1

    def neighbors(self, node):
        if node not in self.nodes:
            raise KeyError(f"node {node} not in graph")
        yield from self.nodes[node].keys()

    def has_node(self, node):
        return node in self.nodes

    def has_edge(self, src, dst):
        if not self.has_node(src):
            return False
        else:
            return dst in self.nodes[src]

    def self_loop(self):
        loop_edges = []
        for u in self.nodes:
            for v in self.nodes[u]:
                if v == u:
                    loop_edges.append(u)
        return loop_edges

    @property
    def edges(self):
        t = []
        for u in self.nodes:
            for v in self.neighbors(u):
                t.append((u, v))
        yield from t

    def prim(self) -> Edges:
        """Find a minimum spanning tree of a graph g using Prim's algorithm.
            v.key = min {w(u,v) | u in S}

            Running Time: O(E lg V) using a binary heap."""

        V = self.nodes
        s = V.pop()
        Q = [(0, s)]

        key = {node: self.INF for node in V}
        key[s] = 0
        parent = {v: None for v in self.nodes}

        S = set()

        while Q:  # |Q| = |V|
            u = heappop(Q)[1]  # O (lg |V|)
            S.add(u)
            for v in self.neighbors(u):  # O(deg[v]) ... ∑ (deg(v)) ∀ v ⊆ V = O(|E|)
                if v not in S and self.weight(u, v) < key[v]:  # O(1)
                    key[v] = self.weight(u, v)
                    heappush(Q, (key[v], v))  # o(lg |V|)
                    parent[v] = u

        mst = [(v, parent[v]) for v in V]
        return mst

    def kruskal(self) -> Edges:
        """Kruskal's algorithm."""

        V = self.nodes
        T = UnionFind([v for v in V])  # O(|V|) make-set() calls
        # O(|E| lg |E|) or O(|E|) when using counting sort if weights are integer weights in the range O(|E|^O(1))
        E = sorted(self.edges, key=lambda x: self.weight(*x))
        mst = []

        for u, v in E:  # O(|E|)
            if T[u] != T[v]:  # amortized O(⍺(V))
                mst.append((u, v))
                T.union(u, v)
        return mst

    def __contains__(self, item):
        if len(item) == 1:  # Assume this is a node
            return self.has_node(item)
        else:
            return self.has_edge(item)

    def __repr__(self):
        return repr(self.nodes)

    @staticmethod
    def insert_supersource(g, sources=None):
        if type(g.nodes) != dict:
            raise TypeError('this method works with dictionary representations of static graphs')
        if sources is None:
            sources = g.nodes.keys()
        g.nodes[Graph.supersource] = {source: 0 for source in sources}

    def bellman_ford(self, source, return_type=None):
        distances = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        distances[source] = 0

        for i in range(len(self.nodes) - 1):  # O(|V| - 1)
            for u in self.nodes:  # O(|E|)
                for v in self.neighbors(u):
                    if distances[u] + self.weight(u, v) < distances[v]:  # O(1)
                        distances[v] = self.weight(u, v) + distances[u]
                        pred[v] = u

        for u in self.nodes:  # checking for negative-weight edge cycles
            for v in self.neighbors(u):
                if distances[u] + self.weight(u, v) < distances[v]:
                    print("the input graph contains a negative-weight cycle")
                    return False, None

        return True, Graph.return_data(locals(), source, pred, distances, return_type)

    @staticmethod
    def return_data(locals_dict, source, pred, distances, return_iterable):
        """Private"""
        # return stuff
        if type(return_iterable) == str:
            if return_iterable not in locals_dict:
                raise KeyError('possible options are pred, distances, and None for printing path')
            else:
                return locals_dict[return_iterable]

        if return_iterable is None:
            return Graph.get_path(distances, source, pred)
        else:
            l = []
            for ret in return_iterable:
                if ret is None:
                    Graph.get_path(distances, source, pred)
                else:
                    if ret not in locals():
                        raise KeyError('possible options are pred, distances, and None for printing path')
                    else:
                        l.append(ret)
            return tuple(l)

    @staticmethod
    def get_path(distances, source, pred):
        paths = []
        for node in pred:
            if node != source:
                curr = node
                temp = []
                while curr is not None and curr != source:
                    temp.append(curr)
                    curr = pred[curr]
                temp.append(source)
                temp.reverse()
                if curr is None:
                    paths.append('No path from ' + str(source) + ' to ' + str(node))
                else:
                    paths.append('the path from ' + str(source) + ' to ' + str(node) + ' is: ' + '->'.join(temp) +
                                 ' and the weight is ' + ' : ' + str(distances[node]))
        return paths

    @property
    def is_empty(self):
        return len(self.nodes) == 0

    def dijkstra(self, source, w=None, target=None, return_type=None):
        """Neat implementation of dijkstra"""
        w = self.weight if w is None else w  # use the default weight function
        distances = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        distances[source] = 0
        Q = [(0, source)]

        while Q:
            u = heappop(Q)[1]  # extract_min
            if u == target:
                break
            for v in self.neighbors(u):
                if distances[u] + w(u, v) < distances[v]:  # relaxation
                    distances[v] = distances[u] + w(u, v)
                    pred[v] = u
                    heappush(Q, (distances[v], v, u))

        return Graph.return_data(locals(), source, pred, distances, return_type)

    def pop_supersource(self, D):
        for u in D:
            D[u].pop(self.supersource)
        D.pop(self.supersource)

    def floyd_warshall(self, path=False):
        if path:
            return self.__floyd_warshall_with_path()
        else:
            return self.__floyd_warshall_without_path()

    def __floyd_warshall_with_path(self):
        """All pairs shortest path algorithm
        Better for dense graphs"""
        c = count(0)
        key = {v: next(c) for v in self.nodes}
        n = len(key)

        dist = np.full((n, n), inf)
        succ = np.ful((n, n), -1)
        for (u, v) in self.edges:
            dist[key[u]][key[v]] = self.weight(u, v)
            succ[key[u]][key[v]] = v
        for v in self.nodes:
            dist[key[v]][key[v]] = 0
            succ[key[v]][key[v]] = v

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        succ[i][j] = succ[i][k]

        D = defaultdict(dict)
        for u in self.nodes:
            for v in self.nodes:
                D[u][v] = dist[key[u]][key[v]]
        return dict(D), succ

    def __floyd_warshall_without_path(self):
        """All pairs shortest path algorithm
        Better for dense graphs"""
        c = count(0)
        key = {v: next(c) for v in self.nodes}
        n = len(key)

        dist = np.full((n, n), inf)
        for (u, v) in self.edges:
            dist[key[u]][key[v]] = self.weight(u, v)
        for v in self.nodes:
            dist[key[v]][key[v]] = 0

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        D = {v: {} for v in self.nodes}
        for u in self.nodes:
            for v in self.nodes:
                D[u][v] = dist[key[u]][key[v]]
        return D

    @staticmethod
    def reconstruct_path(target, pred) -> Path:
        path = [target]
        current = pred[target]
        while current is not None:
            path.append(current)
            current = pred[current]
        return list(reversed(path))

    def a_star(self, source: Node, target: Node, h: FunctionType = lambda x: 0) -> Path:
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

        pred = defaultdict(lambda: None)

        g_score = defaultdict(lambda: self.INF)
        g_score[source] = 0

        f_score = defaultdict(lambda: self.INF)
        f_score[source] = h(source)

        Q = [(f_score[source], source)]
        fringe = {source}  # openSet

        while Q:
            u = heappop(Q)[1]
            if u == target:
                return self.reconstruct_path(u, pred)
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

        # goal was never reached
        print("Goal was never reached.")
        return []

    def johnsons(self):
        """All-pairs shortest paths"""
        g = deepcopy(self.nodes)  # we dont want to modify the original graph
        Graph.insert_supersource(self)

        path_exists, h = self.bellman_ford(Graph.supersource, 'distances')
        if not path_exists:
            return False
        else:
            w_hat = defaultdict(dict)
            for u, v in self.edges:
                w_hat[u][v] = self.weight(u, v) + h[u] - h[v]

            D = defaultdict(dict)
            for u in self.nodes:
                du_prime = self.dijkstra(u, w=lambda x, y: w_hat[x][y], return_type='distances')
                for v in self.nodes:
                    D[u][v] = du_prime[v] + h[v] - h[u]
            self.nodes = g
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
        """ Iterative deepening depth first search"""
        pass

    def euler_tour(self, source):
        """Works for DAGS"""
        visited = set()
        order = []
        seen = set()

        def euler_visit(u, order):
            order.append(u)
            seen.add(u)
            for v in self.neighbors(u):
                if v not in visited and v not in seen:
                    euler_visit(v, order)
            visited.add(u)
            if len(list(self.neighbors(u))) != 0:
                order.append(u)

        euler_visit(source, order)
        pprint('->'.join(order))
        return order

    def iterative_dfs(self, s):
        """Stack based
        Buggy. Assignment: Topological sort using Iterative DFS"""
        # TODO: implement

    def korasaju(self):
        # TODO: implement
        pass

    def a_star(self):
        # TODO: implement
        pass

    def kahn_topsort(self):
        # TODO: implement
        pass

    def dfs(self, source, sort=True):
        """Recursive dfs"""
        visited = set()
        d = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        d[source] = 0

        def visit(u, time, top_sort):
            time += 1
            d[u] = time
            for v in self.neighbors(u):
                if v not in visited and d[v] == inf:
                    pred[v] = u
                    visit(v, time, top_sort)
            visited.add(u)

            if top_sort:
                top_sort.appendleft(u)

        top_sort = None if not sort else deque()
        for u in self.nodes:
            if u not in visited:
                visit(u, sort, top_sort)
        return top_sort, pred
