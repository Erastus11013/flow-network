from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from heapq import heappop, heappush
from math import inf
from pprint import pprint
from random import choice
from sys import maxsize
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    TypeVar,
    Union,
)
from uuid import uuid4

import graphviz
import numpy as np
from more_itertools import first

INF = 1 << 64


@dataclass(frozen=True, slots=True)
class Node:
    id: int | str
    attributes: dict[str, object] = field(default_factory=dict, hash=False)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __add__(self, other):
        return Node(
            str(self.id) + str(other.id), {**self.attributes, **other.attributes}
        )


super_source = Node(-maxsize)
super_sink = Node(maxsize)


@dataclass(slots=True, eq=True)
class FlowNetworkEdgeAttributes:
    cap: int
    flow: int = 0
    reversed: bool = False


FlowNetworkEdge = Tuple[Node, Node, FlowNetworkEdgeAttributes]
FlowNetworkEdges = List[FlowNetworkEdge]
Path = List[Node]
Distances = dict[Node, float]


class Predecessors(dict[Node, Optional[Node]]):
    def reset(self, nodes: Iterable[Node], source: Node) -> None:
        for node in nodes:
            self[node] = None
        self[source] = source


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


def _gen_paths_str(distances: Distances, source: Node, pred: Predecessors) -> list[str]:
    paths = []
    for sink in pred:
        if sink == source:
            continue
        path = _gen_path(source, sink, pred)
        if path and path[0] == source:
            paths.append(
                "the path from "
                + str(source)
                + " to "
                + str(sink)
                + " is: "
                + "->".join(map(str, path))
                + " and the weight is "
                + " : "
                + str(distances[sink])
            )
        else:
            paths.append("No path from " + str(source) + " to " + str(sink))
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
            return _gen_paths_str(distances, source, pred)
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


T = TypeVar("T")


class Graph(defaultdict[Node, dict[Node, T]], Generic[T], ABC):
    def adjacency(self, src: Node) -> Iterable[Node]:
        yield from self[src]

    def adjacency_with_attrs(self, src: Node) -> Iterable[tuple[Node, T]]:
        for dst, edge_attributes in self[src].items():
            yield dst, edge_attributes

    @property
    def edges(self) -> Iterable[tuple[Node, Node]]:
        for src in self:
            for dst in self[src]:
                yield src, dst

    @property
    def edges_with_attrs(self) -> Iterable[tuple[Node, Node, T]]:
        for src in self:
            for dst, edge_attributes in self[src].items():
                yield src, dst, edge_attributes

    @property
    def nodes(self) -> Iterator[Node]:
        yield from self.keys()

    @property
    def n_nodes(self) -> int:
        return len(self)

    def n_edges(self):
        return sum(len(self[src]) for src in self)

    def __contains__(self, item) -> bool:
        match item:
            case Node() as node:
                return super().__contains__(node)
            case (Node() as src, Node() as dst):
                if src not in self:
                    return False
                return dst in self[src]
            case _:
                raise ValueError("can only search nodes and edges")

    def random_node(self):
        return choice(tuple(self.keys()))

    def remove_anti_parallel_edges(self):
        ap_edges = []
        for u, v in self.edges:
            if u in self[v]:
                ap_edges.append((u, v))
        for u, v in ap_edges:
            v_prime = Node(
                str(u) + str(v)
            )  # create new label by concatenating original labels
            cap_flow_st = self[u].pop(v)
            self[u][v_prime] = cap_flow_st
            self[v_prime][v] = cap_flow_st
        return True

    def remove_self_loops(self) -> None:
        s_loop = []
        for u in self:
            for v in self[u]:
                if v == u:
                    s_loop.append(u)
        for u in s_loop:
            self[u].pop(u)

    def get_self_loop_nodes(self):
        self_loop_nodes = []
        for u in self:
            if u in self[u]:
                self_loop_nodes.append(u)
        return self_loop_nodes

    def bfs(self, s: Node, target: Optional[Node] = None) -> Distances:
        distance: Distances = defaultdict(int)
        Q = deque([s])
        while Q:
            u = Q.pop()
            if u == target:
                return distance
            for v in self.adjacency(u):
                if not distance[v]:
                    distance[v] = distance[u] + 1
                    Q.appendleft(v)
        return distance

    @abstractmethod
    def version(self) -> int:
        ...


class ShallowCopy(Graph[float]):
    def __init__(self):
        super().__init__(dict)

    def version(self):
        return id(self)


class Digraph(Graph[T], ABC):
    def insert_edge(self, src: Node, dst: Node, attr: T) -> None:
        if dst not in self:
            self[dst] = {}
        if src not in self:
            self[src] = {}
        self[src][dst] = attr

    def insert_edges_from_iterable(self, edges: Iterable[tuple[Node, Node, T]]) -> None:
        """Assumes that the nodes are in the order.
        (src, dst, attributes)
        """
        for src, dst, attrs in edges:
            self.insert_edge(src, dst, attrs)

    @abstractmethod
    def weight(self, u: Node, v: Node) -> float:
        ...

    def reversed(self):
        rev_g = self.__new__(self.__class__)
        for edge in self.edges:
            src, dst, *rest = edge
            rev_g.insert_edge(dst, src, *rest)
        return rev_g

    def bellman_ford(
        self,
        source: Node,
        return_type_option: ReturnTypeOption,
        sink: Optional[Node] = None,
    ) -> ReturnType:
        distances: Distances = defaultdict(lambda: INF)
        pred: Predecessors = Predecessors()
        distances[source] = 0

        for i in range(len(self) - 1):  # O(|V| - 1)
            for u in self:  # O(|E|)
                for v in self.adjacency(u):
                    if distances[u] + self.weight(u, v) < distances[v]:  # O(1)
                        distances[v] = self.weight(u, v) + distances[u]
                        pred[v] = u

        for u in self:  # checking for negative-weight edge cycles
            for v in self.adjacency(u):
                if distances[u] + self.weight(u, v) < distances[v]:
                    raise ValueError("the input graph contains a negative-weight cycle")
        return _resolve_return_type(return_type_option, distances, pred, source, sink)

    def dijkstra(
        self,
        source: Node,
        return_type_option: ReturnTypeOption,
        w: Optional[Callable[[Node, Node], float]] = None,
        target: Optional[Node] = None,
    ):
        """Neat implementation of dijkstra"""
        w = self.weight if w is None else w  # use the default weight function
        distances: Distances = defaultdict(lambda: INF)
        pred: Predecessors = Predecessors()
        distances[source] = 0
        Q: list[tuple[float, Node]] = [(0, source)]

        while Q:
            _, u = heappop(Q)  # extract_min
            if u == target:
                break
            for v in self.adjacency(u):
                if distances[u] + w(u, v) < distances[v]:  # relaxation
                    distances[v] = distances[u] + w(u, v)
                    pred[v] = u
                    heappush(Q, (distances[v], v))

        return _resolve_return_type(return_type_option, distances, pred, source, target)

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

        Dijkstra can be viewed as a special case of A* where h(n) = 0, because it is a greedy algorithm.
        It makes the best
        choice locally with no regards to the future"""

        assert self, "Empty graph."
        assert source in self, f"Missing source {source}."
        assert target in self, f"Missing target {target}."

        pred: Predecessors = Predecessors()

        g_score: Distances = defaultdict(lambda: INF)
        g_score[source] = 0

        f_score: Distances = defaultdict(lambda: INF)
        f_score[source] = h(source)

        Q: list[tuple[float, Node]] = [(f_score[source], source)]
        fringe = {source}  # openSet

        while Q:
            _, u = heappop(Q)
            if u == target:
                return _resolve_return_type(return_type, g_score, pred, source, target)
            fringe.remove(u)
            for v in self.adjacency(u):
                temp_g = g_score[u] + self.weight(u, v)
                if temp_g < g_score[v]:
                    pred[v] = u
                    g_score[v] = temp_g
                    f_score[v] = g_score[v] + h(v)
                    if v not in fringe:
                        fringe.add(v)
                        heappush(Q, (f_score[v], v))

        raise ValueError("Goal was never reached.")

    @staticmethod
    def insert_super_source(g):
        if type(g) != dict:
            raise TypeError(
                "this method works with dictionary representations of static graphs"
            )
        g[super_source] = defaultdict(int)

    @staticmethod
    def pop_super_source(D):
        for u in D:
            D[u].pop(super_source)
        D.pop(super_source)

    def johnsons(self):
        """All-pairs shortest paths"""
        g = deepcopy(self)  # we don't want to modify the original graph
        g.insert_super_source(g)

        path_exists, h = self.bellman_ford(
            super_source, return_type_option=ReturnTypeOption.EXISTS
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
            self.pop_super_source(D)
            return dict(D)

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

        for i in self.nodes:
            for j in self.nodes:
                d[i][j] = self.weight(i, j)

        for i in self.nodes:
            d[i][i] = 0

        for k in self.nodes:
            for j in self.nodes:
                for i in self.nodes:
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

        for i in self.nodes:
            for j in self.nodes:
                d[i][j] = self.weight(i, j)
                if d[i][j] != INF:
                    nxt[i][j] = j
                else:
                    nxt[i][j] = None

        for i in self.nodes:
            d[i][i] = 0

        for k in self.nodes:
            for j in self.nodes:
                for i in self.nodes:
                    x = d[i][j]
                    y = d[i][k] + d[k][j]
                    if y < x:
                        d[i][j] = y
                        nxt[i][j] = nxt[i][k]
        return d, nxt

    def num_neighbors(self, node: Node) -> int:
        return len(self[node])

    def korasaju_scc(self):
        times, _ = self.iterative_dfs(self.nodes, self)
        _, components = self.iterative_dfs(reversed(times), self.reversed())
        return components

    def topsort(self):
        times, _ = self.iterative_dfs(self.nodes, self)
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
                S.extend((False, v) for v in g.adjacency(u))

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
        queue = deque([v for v in self.nodes if not in_deg[v]])
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
        d = defaultdict(lambda: INF)
        pred = defaultdict(lambda: None)
        d[source] = 0

        def visit(cur_node, time, top):
            time += 1
            d[cur_node] = time
            for v in self.adjacency(cur_node):
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
            for v in self.adjacency(u):
                if v not in visited and v not in seen:
                    euler_visit(v, order_)
            visited.add(u)
            if len(list(self.adjacency(u))) != 0:
                order_.append(u)

        euler_visit(source, order)
        if print_path:
            pprint("->".join(order))
        return order

    def shallow_reverse(self) -> ShallowCopy:
        rev_g = ShallowCopy()
        for u, v in self.edges:
            rev_g[v][u] = 0
        return rev_g

    def is_bipartite(self) -> bool:
        """A bipartite graph (or bigraph) is a graph whose vertices can be divided into two disjoint
            and independent sets U and V such that every edge connects a vertex in U to one in V.
            Vertex sets U and V are usually called the parts of the graph.
            Equivalently, a bipartite graph is a graph that does not contain any odd-length cycles.

        Let G be a graph. Then G is 2-colorable if and only if G is bipartite.
        source: https://cp-algorithms.com/graph/bipartite-check.html
        """

        color: dict[Node, int] = defaultdict(lambda: -1)  # Literal[-1, 0, 1]
        Q: deque[Node] = deque()
        is_bipartite = True

        for source in self.nodes:
            if color[source] == -1:
                Q.appendleft(source)
                if self.num_neighbors(source) == 0:
                    continue
                color[source] = 0
                while Q:
                    v = Q.pop()
                    for u in self.adjacency(v):
                        if color[u] == -1:
                            color[u] = color[v] ^ 1
                            Q.appendleft(u)
                        else:
                            is_bipartite &= color[u] != color[v]
        return is_bipartite


class FlowNetwork(Digraph[FlowNetworkEdgeAttributes]):
    def __init__(self):
        super().__init__()
        self._version = uuid4().int

    def version(self) -> int:
        return self._version

    def weight(self, u: Node, v: Node) -> float:
        raise NotImplementedError("flow network edges have undefined weight")

    def insert_edge(self, src: Node, dst: Node, *args, **kwargs) -> None:
        super().insert_edge(src, dst, FlowNetworkEdgeAttributes(*args, **kwargs))
        self._version = uuid4().int

    def insert_edges_from_iterable(
        self, edges: Iterable[tuple[Node, Node, Any]]
    ) -> None:
        """Assumes that the nodes are in the order.
        (src, dst, attributes)
        """
        for src, dst, *rest in edges:
            self.insert_edge(src, dst, *rest)

    def capacity(self, u, v) -> int:
        return self[u][v].cap

    def flow(self, u, v) -> int:
        return self[u][v].flow

    def residual_capacity(self, u, v) -> int:
        return self[u][v].cap - self[u][v].flow

    def set_weights(self, edges, weights):
        if len(edges) != len(weights):
            raise ValueError("number of edges must equal number of weights")
        for edge, weight in zip(edges, weights):
            src, dst = edge
            self[src][dst].weight = weight

    def maxflow(self, source):
        val_f = 0
        for v in self[source]:
            val_f += self[source][v].flow
        return val_f

    def initialize_reversed_edges(self):
        for u, v, edge_attribute in self.edges_with_attrs:
            if edge_attribute.reversed:
                assert self[v][u].cap == edge_attribute.cap and not self[v][u].reversed
                continue
            c = self[u][v].cap
            self[v][u] = FlowNetworkEdgeAttributes(
                c, self.residual_capacity(u, v), reversed=True
            )

    def set_flows(self, val):
        for u in self:
            for v in self[u]:
                self[u][v].flow = val

    def multiple_max_flow(self, sources: Iterable, sinks: Iterable, cap=1):
        self[super_source] = {
            source: FlowNetworkEdgeAttributes(cap) for source in sources
        }  # flow is 0
        for sink in sinks:
            self[sink][super_sink].cap = cap
        return True

    def check_capacity_constraints(self):
        for u, v in self.edges:
            if not (0 < self.flow(u, v) <= self.capacity(u, v)):
                raise ValueError(f"capacity constraint violated on edge ({u}, {v})")

    def check_flow_conservation(self, source: Node, sink: Node):
        incoming = self.shallow_reverse()
        outgoing = self
        for node in self:
            if node == source or node == sink:
                continue
            if sum(self.flow(node, v) for v in outgoing[node]) != sum(
                self.flow(u, node) for u in incoming[node]
            ):
                raise ValueError(f"flow conservation violated at node {node}")
        if not sum(self.flow(source, v) for v in outgoing[source]) == sum(
            self.flow(u, sink) for u in incoming[sink]
        ):
            raise ValueError(
                f"flow conservation violated at source {source} or sink {sink}"
            )

    def copy(self: Self) -> Self:
        flow_copy = FlowNetwork()
        for u, v, attrs in self.edges_with_attrs:
            flow_copy.insert_edge(u, v, attrs.cap, attrs.flow, attrs.reversed)
        return flow_copy

    def dot(self, src: Node, dst: Node):
        dot = graphviz.Digraph(
            f"flow network {id(self)}",
            format="pdf",
            graph_attr={"rankdir": "LR", "fontname": "Courier"},
        )
        for node in self.nodes:
            if node == src:
                dot.node(str(node.id), color="green")
            elif node == dst:
                dot.node(str(node.id), color="red")
            else:
                dot.node(str(node.id))
        for u, v, attrs in self.edges_with_attrs:
            if attrs.reversed:
                dot.edge(str(u.id), str(v.id), f"{attrs.flow}/{attrs.cap}", color="red")
            else:
                dot.edge(str(u.id), str(v.id), f"{attrs.flow}/{attrs.cap}")
        return dot

    def __hash__(self):
        return self._version

    @cache
    def ordered_neighbors_list(self, node: Node) -> list[Node]:
        return list(self[node].keys())
