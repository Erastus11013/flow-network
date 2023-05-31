from collections import deque

from core import Predecessors, Node
from edmonds_karp import FlowNetwork, defaultdict


class CapacityScaler(FlowNetwork):
    __slots__ = "max_capacity"

    def __init__(self):
        super().__init__()

        self.max_capacity = -self.INF

    def insert_edges_from_iterable(self, edges):
        for edge in edges:
            self.insert_edge(edge)
            self.max_capacity = max(self.max_capacity, self[edge[0]][edge[1]].cap)

    def _bfs_capacity_scaling(
        self, parent: Predecessors, source: Node, sink: Node, delta: int
    ) -> int:
        queue = deque()
        queue.append((source, self.INF))

        for node in self:
            parent[node] = None
        parent[source] = source

        while queue:
            current, flow = queue.popleft()

            for nxt in self.neighbors(current):
                if (
                    parent[nxt] is None
                    and self[current][nxt].cap >= delta
                    and self[current][nxt].cap - self[current][nxt].flow > 0
                ):
                    parent[nxt] = current
                    bottleneck = min(
                        flow, self[current][nxt].cap - self[current][nxt].flow
                    )
                    if nxt == sink:
                        return bottleneck
                    queue.append((nxt, bottleneck))

        return 0

    def augment_paths(self, source, sink, delta):
        """Find all augmenting paths with a bottleneck capacity >= delta."""

        parent = defaultdict(lambda: None)
        visited = set()
        stack = [(source, self.INF)]
        while stack:
            u, df = stack.pop()

            if u == sink:
                current = sink
                while current != source:
                    p = parent[current]
                    self[p][current].flow += df
                    self[current][p].flow -= df
                    current = p
                continue

            if u in visited:
                continue

            visited.add(u)
            for v in self[u]:
                if (self[u][v].cap - self[u][v].flow) >= delta and v not in visited:
                    parent[v] = u
                    stack.append((v, min(df, self[u][v].cap - self[u][v].flow)))

    def find_max_flow(self, source, sink):
        self.set_flows(0)
        self.init_reversed_edges()

        delta = (
            1 << (self.max_capacity - 1).bit_length()
        )  # smallest power of 2 greater than or equal to U
        parent = Predecessors()
        while delta >= 1:
            while flow := self._bfs_capacity_scaling(parent, source, sink, delta):
                v = sink
                while v != source:
                    u = parent[v]
                    self[u][v].flow += flow
                    self[v][u].flow -= flow
                    v = u
            delta >>= 1
        return self.maxflow(source)
