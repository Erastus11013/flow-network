from abc import ABC, abstractmethod
from collections import deque, defaultdict
from heapq import heappush, heappop

from core import FlowNetwork, Predecessors, Node


class MaxFlowSolver(ABC):
    def __init__(self, graph: FlowNetwork):
        self.graph = graph
        self.graph.set_flows(0)
        self.graph.init_reversed_edges()

    @abstractmethod
    def solve(self, source, sink):
        pass


class EdmondsKarpSolver(MaxFlowSolver):
    def solve(self, source: Node, sink: Node):
        """Edmonds Karp implementation of the Ford Fulkerson method
        Notice:
        if graph may have some antiparallel edges:
            add this line: self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        if graph may have self-loops:
            add this line: self.remove_self_loops()
        """
        parent = Predecessors()
        max_flow = 0

        while (flow := self._bfs(parent, source, sink)) != 0:
            max_flow += flow
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v].flow += flow
                self.graph[v][u].flow -= flow
                v = u
        return max_flow

    def _bfs(self, parent: Predecessors, source: Node, sink: Node) -> int:
        queue = deque()
        queue.append((source, self.graph.INF))

        for node in self.graph:
            parent[node] = None
        parent[source] = source

        while queue:
            current, flow = queue.popleft()

            for nxt in self.graph.neighbors(current):
                if (
                    parent[nxt] is None
                    and self.graph[current][nxt].cap - self.graph[current][nxt].flow > 0
                ):
                    parent[nxt] = current
                    bottleneck = min(
                        flow,
                        self.graph[current][nxt].cap - self.graph[current][nxt].flow,
                    )
                    if nxt == sink:
                        return bottleneck
                    queue.append((nxt, bottleneck))

        return 0


class CapacityScalingSolver(MaxFlowSolver):
    def _bfs_capacity_scaling(
        self, parent: Predecessors, source: Node, sink: Node, delta: int
    ) -> int:
        queue = deque()
        queue.append((source, self.graph.INF))

        for node in self.graph:
            parent[node] = None
        parent[source] = source

        while queue:
            current, flow = queue.popleft()

            for nxt in self.graph.neighbors(current):
                if (
                    parent[nxt] is None
                    and self.graph[current][nxt].cap >= delta
                    and self.graph[current][nxt].cap - self.graph[current][nxt].flow > 0
                ):
                    parent[nxt] = current
                    bottleneck = min(
                        flow,
                        self.graph[current][nxt].cap - self.graph[current][nxt].flow,
                    )
                    if nxt == sink:
                        return bottleneck
                    queue.append((nxt, bottleneck))

        return 0

    def augment_paths(self, source: Node, sink: Node, delta: int):
        """Find all augmenting paths with a bottleneck capacity >= delta."""

        parent: Predecessors = defaultdict(lambda: None)
        visited = set()
        stack = [(source, self.graph.INF)]
        while stack:
            u, df = stack.pop()

            if u == sink:
                current = sink
                while current != source:
                    p = parent[current]
                    self.graph[p][current].flow += df
                    self.graph[current][p].flow -= df
                    current = p
                continue

            if u in visited:
                continue

            visited.add(u)
            for v in self.graph[u]:
                if (
                    self.graph[u][v].cap - self.graph[u][v].flow
                ) >= delta and v not in visited:
                    parent[v] = u
                    stack.append(
                        (v, min(df, self.graph[u][v].cap - self.graph[u][v].flow))
                    )

    def solve(self, source: Node, sink: Node):
        max_capacity = max(
            edge_attribute.cap for _, _, edge_attribute in self.graph.edges
        )
        delta = (
            1 << (max_capacity - 1).bit_length()
        )  # smallest power of 2 greater than or equal to U
        parent = Predecessors()
        while delta >= 1:
            while flow := self._bfs_capacity_scaling(parent, source, sink, delta):
                v = sink
                while v != source:
                    u = parent[v]
                    self.graph[u][v].flow += flow
                    self.graph[v][u].flow -= flow
                    v = u
            delta >>= 1
        return self.graph.maxflow(source)


class DinicsSolver(MaxFlowSolver):
    def gen_levels(self, source: Node, sink: Node):
        """Creates a layered graph/ admissible graph using breadth first search
        Variables:
            delta: the level of the sink
            lid: the level id
        """

        queue = deque([source])
        levels = {node: 0 for node in self.graph}
        levels[source] = 1

        while queue:
            u = queue.popleft()
            if u == sink:
                break
            for v in self.graph.neighbors(u):
                if (self.graph[u][v].cap - self.graph[u][v].flow) > 0 and levels[
                    v
                ] == 0:
                    levels[v] = levels[u] + 1
                    queue.append(v)

        return levels

    def gen_blocking_flow(self, u, pushed, sink, levels, visited):
        if pushed == 0 or u == sink:
            return pushed
        for v in self.graph.neighbors(u):
            if (
                v not in visited[u]
                and levels[v] == levels[u] + 1
                and self.graph[u][v].cap - self.graph[u][v].flow > 0
            ):
                visited[u].add(v)
                cf = self.gen_blocking_flow(
                    v,
                    min(pushed, self.graph[u][v].cap - self.graph[u][v].flow),
                    sink,
                    levels,
                    visited,
                )
                if cf > 0:
                    self.graph[u][v].flow += cf
                    self.graph[v][u].flow -= cf
                    return cf
        return 0

    def solve(self, source: Node, sink: Node):
        assert source != sink
        max_flow = 0

        while True:
            levels = self.gen_levels(source, sink)
            visited = defaultdict(set)
            if levels[sink] == 0:
                break
            while (
                blocking_flow := self.gen_blocking_flow(
                    source, self.graph.INF, sink, levels, visited
                )
            ) != 0:
                max_flow += blocking_flow
        return max_flow


class PushRelabelSolver(MaxFlowSolver, ABC):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.excess: dict[Node, int] = defaultdict(int)
        self.height: dict[Node, float] = {}

    def initialize_pre_flow(self, source: Node, sink: Node) -> None:
        assert source in self
        # Initialize heights as the shortest distance from the sink to every node except the source
        # We perform bfs on the original graph, not the residual one

        self.height = self.graph.shallow_reverse().bfs(sink)
        self.height[source] = self.graph.num_nodes()

        for u, v, _ in self.graph.edges:
            self.graph[u][v].flow = 0

        # saturate all the edges coming out of the source
        for v, _ in self.graph.neighbors(source):
            self.excess[v] = self.graph[source][v].flow = self.graph[source][v].cap
            self.excess[source] -= self.graph[source][v].cap

        for u, v, _ in tuple(self.graph.edges):
            self.graph.insert_edge(
                v, u, self.graph[u][v].cap, self.graph.residual_capacity(u, v)
            )

        # Create the residual graph of g, Q contains the nodes with positive excesses
        # Initially, those are just the edges outgoing from the source

    def push(self, u: Node, v: Node):
        """Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
        from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
            by δ. If δ = cf (uv), this is called a saturating push, else it is a non-saturating push.
        """
        # assert excess[u] > 0 and height[u] == height[v] + 1
        delta = min((self.graph[u][v].cap - self.graph[u][v].flow), self.excess[u])
        self.graph[u][v].flow += delta
        self.graph[v][u].flow -= delta
        self.excess[u] -= delta
        self.excess[v] += delta

    def relabel(self, u: Node):
        valid = [
            v
            for v, _ in self.graph.neighbors(u)
            if self.graph.residual_capacity(u, v) > 0
        ]
        # assert (len(valid) != 0)
        # assert excess[u] > 0 and all(height[u] <= height[v] for v in valid)
        self.height[u] = min(self.height[v] for v in valid) + 1


class RelabelToFrontSolver(PushRelabelSolver):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.seen: dict[Node, int] = defaultdict(int)

    def discharge(self, u: Node):
        neighbors = tuple(self.graph[u])
        while self.excess[u] > 0:
            if self.seen[u] < self.graph.num_neighbors(u):
                v = neighbors[self.seen[u]]
                if (
                    self.graph.residual_capacity(u, v) > 0
                    and self.height[u] == self.height[v] + 1
                ):
                    self.push(u, v)
                else:
                    self.seen[u] += 1
            else:
                self.relabel(u)
                self.seen[u] = 0

    def solve(self, source: Node, sink: Node):
        """Relabel to front algorithm"""

        # list of valid nodes
        L = list(filter(lambda node: node not in (source, sink), self.graph.nodes()))
        self.initialize_pre_flow(source, sink)
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

        return sum(self.graph[source][v].flow for v, _ in self.graph.neighbors(source))


class FifoPushRelabelSolver(PushRelabelSolver):
    def solve(self, source, sink):
        """        FIFO Push/Relabel
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
        self.initialize_pre_flow(source, sink)
        Q: list[tuple[float, int, Node]] = []
        for u, _ in self.graph.neighbors(source):
            queued.add(u)
            heappush(Q, (-self.height[u], u.id, u))

        while Q:
            # highest active node, has the lowest -(height)
            _, _, u = heappop(Q)
            queued.discard(u)
            if u == sink:
                continue
            for v, _ in self.graph.neighbors(u):
                if self.excess[u] == 0:
                    break
                if (
                    self.height[u] == self.height[v] + 1
                    and self.graph.residual_capacity(u, v) > 0
                ):
                    self.push(u, v)
                    if v not in queued and v not in (source, sink):
                        heappush(Q, (-self.height[v], v.id, v))
                        queued.add(v)
            if self.excess[u] > 0:
                self.relabel(u)
                heappush(Q, (-self.height[u], u.id, u))
                queued.add(u)

        return sum(self.graph[source][v].flow for v, _ in self.graph.neighbors(source))
