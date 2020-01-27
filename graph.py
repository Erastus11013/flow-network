# author: Erastus Murungi

from typing import Iterable, Tuple, List, Callable
from heapq import heappush, heappop
from pprint import pprint
from collections import deque, defaultdict
from math import inf
from copy import deepcopy
from random import randint
from string import ascii_lowercase
from copy import deepcopy
from itertools import count
import numpy as np
from sys import maxsize
from union_find import UnionFind
from random import choice


# typing aliases
Node = object
Edge = Tuple[Node, Node]
Edges = List[Edge]
Path = List[Node]


class Graph:
    """Class representing a digraph"""
    supersource = 'S'
    INF = maxsize

    def __init__(self):
        self.nodes = {}

    def add_nodes(self, args):
        for arg in args:
            self.nodes[arg] = {}

    def add_edges(self, args):
        """assumes arguments are tuples in the format (src, dst, weight) """
        if len(args[0]) < 2:
            raise ValueError("Edges must contain at least 2 arguments")
        if len(args[0]) == 2:
            for arg in args:  # (no weights)
                src, dst = arg
                if src not in self.nodes or dst not in self.nodes:
                    raise ValueError("node", str(src), 'or', str(dst), " not in graph")
                self.nodes[src][dst] = 0
        elif len(args[0]) == 3:
            for arg in args:
                src, dst, weight = arg
                if src not in self.nodes:
                    raise ValueError("node", str(src), 'or', str(dst), " not in graph")
                self.nodes[src][dst] = weight
        else:
            for arg in args:
                src, dst, *data = arg
                if src not in self.nodes or dst not in self.nodes:
                    raise ValueError("node", str(src), "or", str(dst), "not in graph")
                self.nodes[src][dst] = data
        return True

    def set_weights(self, edges, weights):
        if len(edges) != len(weights):
            raise ValueError("number of edges must equal number of weights")
        for i, edge in enumerate(edges):
            if edge in edges:
                src, dst = edge
                self.nodes[src][dst] = weights[i]
            else:
                raise ValueError("Edge not in graph. Add edge by calling graph.add_edges()")

    def weight(self, src, dst):
        """Assumes that the first argument is the weight:"""
        if not self.has_edge((src, dst)):
            return self.INF
        y = self.nodes[src][dst]
        if type(y) == tuple:
            return y[0]
        return y

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

    def has_edge(self, edge):
        src, dst, *_ = edge
        if not self.has_node(src):
            return False
        else:
            return dst in self.nodes[src]

    def loop_exists(self):
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
        return t

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
    def add_supersource(g, sources=None):
        if not (hasattr(g, 'nodes')):
            raise ValueError(
                'g is not a graph, or vertices in the graph are not named nodes, i.e self.nodes doesn\'t exist')
        if type(g.nodes) != dict:
            raise TypeError('this method works with dictionary representations of static graphs')
        if sources is None:
            sources = g.nodes.keys()
        else:
            for source in sources:
                if source not in g.nodes:
                    raise KeyError("make sure the source, " + str(source) + " has been added to the graph")
        g.nodes[Graph.supersource] = {source: 0 for source in sources}

    def bellman_ford(self, source, return_type=None):
        distances = {}
        pred = {}
        for node in self.nodes:
            distances[node] = float('inf')
            pred[node] = None

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
        if w is None:
            w = self.weight  # use the default weight function
        distances, pred = {}, {}
        for node in self.nodes:
            distances[node] = inf
            pred[node] = None
        distances[source] = 0

        Q = [(0, source)]

        while Q:
            u = heappop(Q)[1]  # extract_min
            if target is not None:
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

        D = {v: {} for v in self.nodes}
        for u in self.nodes:
            for v in self.nodes:
                D[u][v] = dist[key[u]][key[v]]
        return D, succ

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

    def a_star(self, source: Node, target: Node, h: Callable = lambda x: 0) -> Path:
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
        Graph.add_supersource(self)

        path_exists, h = self.bellman_ford(Graph.supersource, 'distances')
        if not path_exists:
            return False
        else:
            w_hat = {v: {} for v in self.nodes}
            for u, v in self.edges:
                w_hat[u][v] = self.weight(u, v) + h[u] - h[v]

            D = {v: {} for v in self.nodes}
            for u in self.nodes:
                du_prime = self.dijkstra(u, w=lambda x, y: w_hat[x][y], return_type='distances')
                for v in self.nodes:
                    D[u][v] = du_prime[v] + h[v] - h[u]
            self.nodes = g
            self.pop_supersource(D)
            return D

    def bfs(self, source):

        """Queue based bfs"""

        discovered = {}
        pred = {}

        for node in self.nodes:
            discovered[node] = inf
            pred[node] = None
        discovered[source] = 0

        q = deque([source])
        while q:
            u = q.pop()
            for v in self.neighbors(u):
                if discovered[v] == inf:
                    discovered[v] = discovered[u] + 1
                    pred[v] = u
                    q.appendleft(v)

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
        d = {v: inf for v in self.nodes}
        pred = {v: None for v in self.nodes}
        d[source] = 0

        def _dfs_visit(u, time, top_sort):
            time += 1
            d[u] = time
            for v in self.neighbors(u):
                if v not in visited and d[v] == inf:
                    pred[v] = u
                    _dfs_visit(v, time, top_sort)
            visited.add(u)

            if top_sort is not None:
                top_sort.appendleft(u)

        top_sort = None if not sort else deque()
        for u in self.nodes:
            if u not in visited:
                _dfs_visit(u, sort, top_sort)
        return top_sort, pred


class FlowNetwork(Graph):
    supersource = 'S'
    supersink = 'T'

    def __init__(self):
        Graph.__init__(self)
        self.residual_edges = {}
        self.path = set()

    def add_edges(self, args):
        """Add edges to the graph"""
        if len(args) == 0:
            raise ValueError("Cannot add null edges to the graph.")

        elif len(args[0]) == 2:  # assume it has the form (src, dst):
            for arg in args:
                self._set_edges(*arg[:2], 0, 0)
        elif len(args[0]) == 3:  # assume input has the form  (src, dest, cap)
            for arg in args:
                self._set_edges(*arg[:3], 0)
        elif len(args[0]) == 4:  # assume that the tuple has the form (src, dest, capacity, flow)
            for arg in args:
                self._set_edges(*arg)
        else:
            raise ValueError("Insuffient edge values.")

    def _set_edges(self, src, dst, capacity, flow):
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("some nodes are not in the graph")
        self.nodes[src][dst] = (capacity, flow)
        if not self.is_capacity_conserved(src, dst):
            raise ValueError("capacity cannot be less than the flow")

    def remove_anti_parallel_edges(self):
        """Because of RuntimeError: dictionary changed size during iteration,
        we have to modify the dictionary after iteration"""
        ap_edges = []
        for u in self.nodes:
            for v in self.nodes[u]:
                if u in self.nodes[v]:
                    ap_edges.append((u, v))
                    # loop exists:
        for u, v in ap_edges:
            v_prime = str(u) + str(v)  # create new label by concatenating original labels
            self.nodes[v_prime] = {}
            cap_flow = self.nodes[u].pop(v)
            self.nodes[u][v_prime] = cap_flow
            self.nodes[v_prime][v] = cap_flow
        return True

    def remove_self_loops(self):
        s_loop = []
        for u in self.nodes:
            for v in self.nodes[u]:
                if v == u:
                    s_loop.append(u)

        for u in s_loop:
            del self.nodes[u][u]
        return True

    def multiple_max_flow(self, sources: Iterable, sinks: Iterable, cap=inf):
        """Accepts multiple sinks and multiple sources"""
        for source in sources:
            if source not in self.nodes:
                raise KeyError("make sure the source, " + str(source) + " has been added to the graph")
        for sink in sinks:
            if sink not in self.nodes:
                raise KeyError("make sure the sink, " + str(sink) + " has been added to the graph")

        self.nodes[FlowNetwork.supersource] = {source: (cap, 0) for source in sources}  # flow is 0
        for sink in sinks:
            self.nodes[sink][FlowNetwork.supersink] = (cap, 0)
        self.nodes[FlowNetwork.supersink] = {}
        return True

    def create_residual_graph(self, backward_edges=True):
        """"""
        self.residual_edges = {key: {} for key in self.nodes}

        for u in self.nodes:
            for v in self.neighbors(u):
                if self.residual_capacity(u, v) > 0:
                    self.residual_edges[u][v] = (self.residual_capacity(u, v), False)  # the last argument tells whether
                    # the direction of the edge has been reversed
                if backward_edges:
                    if self.flow(u, v) > 0:
                        self.residual_edges[v][u] = (self.flow(u, v), True)

    def parallel_bfs(self, source):
        pass

    def bfs(self, source):
        discovered = {}
        pred = {}

        for node in self.residual_edges:
            discovered[node] = inf
            pred[node] = None
        discovered[source] = 0

        q = deque([source])
        while q:
            u = q.pop()
            for v in self.neighbors_in_residual_graph(u):
                if discovered[v] == inf:
                    discovered[v] = discovered[u] + 1
                    pred[v] = u
                    q.appendleft(v)
        return pred

    @staticmethod
    def print_path(pred, source, sink):
        p = []
        curr = sink
        while curr is not None and curr != source:
            p.append(curr)
            curr = pred[curr]
        p.append(source)
        if curr is None:
            print("No path from source to ", sink)
        else:
            pprint('->'.join(reversed(p)))

    @staticmethod
    def augmenting_path_exists(pred, sink):
        """If the sink has no predecessors, then there is no path from source s to sink t"""
        return not (pred[sink] is None)

    def augmenting_path(self, pred, source, sink):
        """Returns an iterator to help in updating the original graph G
        Returns a list of tuples in the format (source, dest, residual_capacity, flipped)
        (source, dest): an edge in G_f, it might be reversed or not
        flipped: tells whether the edge had been reversed in the original graph G"""

        path = []
        curr = sink
        while curr != source:
            path.append((pred[curr], curr) + self.residual_edges[pred[curr]][curr])
            curr = pred[curr]
        return reversed(path)

    def neighbors_in_residual_graph(self, node):
        return self.residual_edges[node]

    def residual_capacity(self, src, dst):
        """cf(u,v) = c(u,v) - f(u,v)"""
        return self.capacity(src, dst) - self.flow(src, dst)

    def capacity(self, src, dst):
        """Getter method for readability"""
        return self.nodes[src][dst][0]

    def flow(self, src, dst):
        return self.nodes[src][dst][1]

    def weight(self, src, dst):
        """The bfs assumes that every edge has weight 1"""
        return 1

    def get_max_flow(self, source):
        val_f = 0
        for v in self.neighbors(source):
            val_f += self.flow(source, v)
        return val_f

    def is_capacity_conserved(self, src, dst):
        """sanity check"""
        return self.flow(src, dst) <= self.capacity(src, dst)

    def set_flows(self, val):
        for u in self.nodes:
            for v in self.neighbors(u):
                c, _ = self.nodes[u][v]
                self.nodes[u][v] = (c, val)

    def set_caps(self, val):
        for u in self.nodes:
            for v in self.neighbors(u):
                _, f = self.nodes[u][v]
                self.nodes[u][v] = (val, f)

    def update_network_flow(self, path, cf):
        for arg in path:
            src, dst, flow, flipped = arg
            if flipped:
                c, f = self.nodes[dst][src]
                f -= cf
                self.nodes[dst][src] = (c, f)
            else:
                c, f = self.nodes[src][dst]
                f += cf
                self.nodes[src][dst] = (c, f)

    def _augment(self, pred, source, sink, print_path=True) -> Tuple[float, List]:
        """uses the predecessor dictionary to determine a path
        the path is a list of tuples where each tuple is the format
        (source, dest, residual_capacity, is_reversed) """

        path = list(self.augmenting_path(pred, source,
                                         sink))  # it makes a difference that the reversed iterator is converted to a list
        if print_path:
            self.print_path(pred, source, sink)
        cf = min([tup[2] for tup in path])  # tup[2] contains the flow
        self.update_network_flow(path, cf)
        return cf, path

    def edmond_karp(self, source=None, sink=None, print_path=True):
        """Edmond Karp algorithm of the Ford Fulkerson method
        Track tells which graph to print path from node to track
        Can be used for bipartite matching as well"""

        if source is None:
            source = FlowNetwork.supersource
        if sink is None:
            sink = FlowNetwork.supersink
        if source not in self.nodes:  # sanity check
            return None
        self.remove_self_loops()
        total_flow = 0
        self.set_flows(0)  # f[u,v] = 0
        self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        self.create_residual_graph()
        pred = self.bfs(source)  # run bfs once

        while FlowNetwork.augmenting_path_exists(pred, sink):
            cf, _ = self._augment(pred, source, sink, print_path)
            total_flow += cf
            self.create_residual_graph()
            pred = self.bfs(source)

        assert (self.get_max_flow(source) == total_flow)  # sanity check
        return self.get_max_flow(source)


def test_edmond_karps(nodes, edges):
    g1 = FlowNetwork()
    g1.add_nodes(nodes)
    g1.add_edges(edges)
    max_flow_value = g1.edmond_karp('s', 't')
    return max_flow_value


def generate_random_data(n, m, cap_max, source, sink):
    nodes = [''.join([choice(ascii_lowercase) for _ in range(10)]) for _ in range(n)]
    nodes = list(set(nodes))

    s = set()
    edges = []
    for i in range(m):
        n1 = choice(nodes)
        n2 = choice(nodes)
        while n2 != n1:
            n2 = choice(nodes)
        s.add((n1, n2))
    for edge in s:
        cap = randint(0, cap_max)
        edges.append(edge + (cap, 0))

    for i in range(10):  # add outgoing edges to the source
        s1 = set()
        outgoing = choice(nodes)
        while outgoing in s1:
            outgoing = choice(nodes)
        edge = (source, outgoing, randint(0, cap_max), 0)
        edges.append(edge)

    for i in range(10):  # add incoming edges to the sink
        s1 = set()
        incoming = choice(nodes)
        while incoming in s1:
            incoming = choice(nodes)
        edge = (incoming, sink, randint(0, cap_max), 0)
        edges.append(edge)

    pprint(edges)
    nodes = [source] + nodes + [sink]
    return nodes, edges


def test_dfs():
    a, b, c, d, e = 'a', 'b', 'c', 'd', 'e'
    n1 = [a, b, c, d, e]
    e1 = [(a, c, -4), (a, b, 3), (a, e, 8), (b, c, 7), (b, d, 1), (c, d, 6), (d, a, 2), (d, e, -5), (e, b, 4)]
    g1 = Graph()
    g1.add_nodes(n1)
    g1.add_edges(e1)
    el1 = g1.euler_tour('a')
    pprint(el1)
    x1 = g1.iterative_dfs(a)
    print("Iterative DFS", x1)
    x2 = g1.iddfs(a, 3)
    print("IDDFS", x2)
    x3 = g1.dfs(a)
    print("Recursive DFS", x3)


if __name__ == '__main__':
    # n2, e2 = generate_random_data(100, 20000, 60, 's', 't')
    test_dfs()
