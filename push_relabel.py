from abc import ABC
from collections import defaultdict, deque
from functools import reduce
from heapq import heappop, heappush
from operator import add
from typing import Iterable, TypeVar, Generic, Self

import numpy as np

from core import Distances, EdgeAttributes, Node

T = TypeVar("T")


class Graph(defaultdict[Node, dict[Node, T]], Generic[T], ABC):
    def adjacency(self, src: Node) -> Iterable[tuple[Node, T]]:
        for dst, edge_attributes in self[src].items():
            yield dst, edge_attributes

    def edges(self) -> Iterable[tuple[Node, Node, T]]:
        for src in self:
            for dst, edge_attributes in self[src].items():
                yield src, dst, edge_attributes

    def nodes(self) -> set[Node]:
        return {v for u in self for v in self[u]}

    def bfs(self, s: Node) -> Distances:
        distance: Distances = defaultdict(float)
        Q = deque([s])
        while Q:
            u = Q.pop()
            for v, _ in self.adjacency(u):
                if not distance[v]:
                    distance[v] = distance[u] + 1
                    Q.appendleft(v)
        return distance

    def num_neighbors(self, node: Node) -> int:
        return len(self[node])


class ResidualGraph(Graph[float]):
    def __init__(self):
        super().__init__(dict)


class Digraph(Graph[EdgeAttributes]):
    def __init__(self):
        super().__init__(dict)
        self.excess: dict[Node, int] = defaultdict(int)
        self.height: dict[Node, float] = {}
        self.queued: set[Node] = set()
        self.seen: dict[Node, int] = defaultdict(int)

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

    def insert_edges_from_iterable(
        self, edges: Iterable[tuple[Node, Node, ...]]
    ) -> None:
        """Assumes that the nodes are in the order.
        (src, dst, attributes)
        """
        for src, dst, *rest in edges:
            self.insert_edge(src, dst, *rest)

    def num_nodes(self) -> int:
        return len(self.nodes())

    def num_edges(self) -> int:
        return reduce(add, map(lambda n: len(self[n]), self.keys()))

    def insert_edge(self, src: Node, dst: Node, *args, **kwargs) -> None:
        self[src][dst] = EdgeAttributes(*args, **kwargs)

    def assert_edge_in_graph(self, u, v) -> None:
        if not (u, v) in self:
            raise ValueError(f"Edge ({repr(u)}, {repr(v)}) not in graph")

    def weight(self, u, v) -> float:
        self.assert_edge_in_graph(u, v)
        return self[u][v].weight

    def capacity(self, u, v) -> int:
        self.assert_edge_in_graph(u, v)
        return self[u][v].cap

    def flow(self, u, v) -> int:
        self.assert_edge_in_graph(u, v)
        return self[u][v].flow

    def residual_capacity(self, u, v) -> int:
        return self.capacity(u, v) - self.flow(u, v)

    def residual_graph(self) -> ResidualGraph:
        rg = ResidualGraph()
        for u, v, _ in self.edges():
            if (rc := (self[u][v].cap - self[u][v].flow)) != 0:
                rg[u][v] = rc
            rg[v][u] = self.flow(u, v)
        return rg

    def shallow_reverse(self) -> ResidualGraph:
        rev_g = ResidualGraph()
        for u, v, _ in self.edges():
            rev_g[v][u] = 0
        return rev_g

    def initialize_preflow(self, source: Node, sink: Node) -> Self:
        assert source in self
        # Initialize heights as the shortest distance from the sink to every node except the source
        # We perform bfs on the original graph, not the residual one

        self.height = self.shallow_reverse().bfs(sink)
        self.height[source] = self.num_nodes()

        for u, v, _ in self.edges():
            self[u][v].flow = 0

        # saturate all the edges coming out of the source
        for v, _ in self.adjacency(source):
            self.excess[v] = self[source][v].flow = self[source][v].cap
            self.excess[source] -= self[source][v].cap

        for u, v, _ in tuple(self.edges()):
            self.insert_edge(
                v, u, self[u][v].cap, (self[u][v].cap - self[u][v].flow), 1
            )

        # Create the residual graph of g, Q contains the nodes with positive excesses
        # Initially, those are just the edges outgoing from the source
        return self

    def push(self, u: Node, v: Node):
        """Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
        from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
            by δ. If δ = cf (uv), this is called a saturating push, else it is an non-saturating push.
        """
        # assert excess[u] > 0 and height[u] == height[v] + 1
        delta = min((self[u][v].cap - self[u][v].flow), self.excess[u])
        self[u][v].flow += delta
        self[v][u].flow -= delta
        self.excess[u] -= delta
        self.excess[v] += delta

    def relabel(self, u: Node):
        valid = [
            v for v, _ in self.adjacency(u) if (self[u][v].cap - self[u][v].flow) > 0
        ]
        # assert (len(valid) != 0)
        # assert excess[u] > 0 and all(height[u] <= height[v] for v in valid)
        self.height[u] = min(self.height[v] for v in valid) + 1

    def node_is_active(self, u: Node, source: Node, sink: Node) -> bool:
        return u != source and u != sink and self.excess[u] > 0


class FifoPushRelabel(Digraph):
    def run(self, source, sink):
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
        queued = set()
        g = self.initialize_preflow(source, sink)
        Q: list[tuple[float, int, Node]] = []
        for u, _ in self.adjacency(source):
            queued.add(u)
            heappush(Q, (-self.height[u], u.id, u))

        while Q:
            # highest active node, has the lowest -(height)
            _, _, u = heappop(Q)
            queued.discard(u)
            if u == sink:
                continue
            for v, _ in self.adjacency(u):
                if self.excess[u] == 0:
                    break
                if self.height[u] == self.height[v] + 1 and (g[u][v].cap - g[u][v].flow) > 0:
                    self.push(u, v)
                    if v not in queued and v not in (source, sink):
                        heappush(Q, (-self.height[v], v.id, v))
                        queued.add(v)
            if self.excess[u] > 0:
                self.relabel(u)
                heappush(Q, (-self.height[u], u.id, u))
                queued.add(u)

        return sum(g[source][v].flow for v, _ in self.adjacency(source))


class RelabelToFront(Digraph):
    def discharge(self, u: Node):
        neighbors = tuple(self[u])
        while self.excess[u] > 0:
            if self.seen[u] < self.num_neighbors(u):
                v = neighbors[self.seen[u]]
                if (self[u][v].cap - self[u][v].flow) > 0 and self.height[u] == self.height[
                    v
                ] + 1:
                    self.push(u, v)
                else:
                    self.seen[u] += 1
            else:
                self.relabel(u)
                self.seen[u] = 0

    def run(self, source, sink):
        """Relabel to front algorithm"""

        # list of valid nodes
        L = list(filter(lambda node: node not in (source, sink), self.nodes()))
        g = self.initialize_preflow(source, sink)
        p, n = 0, len(L)

        while p < n:
            u = L[p]
            old_height = self.height[u]
            self.discharge(u)
            if self.height[u] > old_height:
                L.insert(0, L.pop(p))  # move to front
                p = 0
            else:
                p += 1

        return sum(g[source][v].flow for v, _ in self.adjacency(source))


def rand_flow_cap(lim: int) -> tuple:
    cap = np.random.randint(2, lim)
    return cap, np.random.randint(1, cap)
