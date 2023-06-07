from abc import ABC, abstractmethod
from collections import defaultdict, deque
from heapq import heappop, heappush

from core import INF, FlowNetwork, Node, Predecessors


class MaxFlowSolver(ABC):
    def __init__(self, graph: FlowNetwork):
        self.graph = graph.copy()
        self.original_graph = graph

    @abstractmethod
    def solve(self, source: Node, sink: Node):
        pass

    def __eq__(self, other):
        return self.original_graph == other.original_graph


class AugmentingPathSolver(MaxFlowSolver, ABC):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.graph.set_flows(0)
        self.graph.init_reversed_edges()

    def has_path(self, source: Node, sink: Node) -> bool:
        distances = self.original_graph.bfs(source, sink)
        if sink not in distances:
            return False
        return True

    def update_path(
        self, source: Node, sink: Node, predecessors: Predecessors, bottleneck: int
    ):
        current = sink
        while current != source:
            pred_current = predecessors[current]
            self.graph[pred_current][current].flow += bottleneck
            self.graph[current][pred_current].flow -= bottleneck
            current = pred_current


class EdmondsKarpSolver(AugmentingPathSolver):
    def solve(self, source: Node, sink: Node):
        """Edmonds Karp implementation of the Ford Fulkerson method
        Notice:
        if graph may have some antiparallel edges:
            add this line: self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        if graph may have self-loops:
            add this line: self.remove_self_loops()
        """
        assert source != sink

        if self.has_path(source, sink):
            predecessors: Predecessors = Predecessors()
            max_flow = 0
            while (
                bottleneck := self.find_augmenting_path(predecessors, source, sink)
            ) != 0:
                self.update_path(source, sink, predecessors, bottleneck)
                max_flow += bottleneck
            return max_flow
        return 0

    def find_augmenting_path(
        self, predecessors: Predecessors, source: Node, sink: Node
    ) -> int:

        queue: deque[tuple[Node, int]] = deque([(source, INF)])
        predecessors.reset(self.graph.nodes, source)

        while queue:
            node, bottleneck = queue.popleft()
            for neighbor in self.graph.adjacency(node):
                if (
                    predecessors[neighbor] is None
                    and self.graph.residual_capacity(node, neighbor) > 0
                ):
                    predecessors[neighbor] = node
                    next_bottleneck = min(
                        bottleneck,
                        self.graph.residual_capacity(node, neighbor),
                    )
                    if neighbor == sink:
                        return next_bottleneck
                    queue.append((neighbor, next_bottleneck))
        return 0


class CapacityScalingSolver(AugmentingPathSolver):
    def find_augmenting_path_delta_residual(
        self, predecessors: Predecessors, source: Node, sink: Node, delta: int
    ) -> int:

        queue: deque[tuple[Node, int]] = deque([(source, INF)])
        predecessors.reset(self.graph.nodes, source)

        while queue:
            node, bottleneck = queue.popleft()
            for neighbor in self.graph.adjacency(node):
                if (
                    predecessors[neighbor] is None
                    and self.graph[node][neighbor].cap >= delta
                    and self.graph.residual_capacity(node, neighbor) > 0
                ):
                    predecessors[neighbor] = node
                    next_bottleneck = min(
                        bottleneck, self.graph.residual_capacity(node, neighbor)
                    )
                    if neighbor == sink:
                        return next_bottleneck
                    queue.append((neighbor, next_bottleneck))
        return 0

    def solve(self, source: Node, sink: Node):
        assert source != sink

        if self.has_path(source, sink):
            max_capacity = max(
                edge_attribute.cap
                for _, _, edge_attribute in self.graph.edges_with_attrs
            )
            delta = (
                1 << (max_capacity - 1).bit_length()
            )  # smallest power of 2 greater than or equal to U
            predecessors: Predecessors = Predecessors()
            while delta >= 1:
                while bottleneck := self.find_augmenting_path_delta_residual(
                    predecessors, source, sink, delta
                ):
                    self.update_path(source, sink, predecessors, bottleneck)
                delta >>= 1
            return self.graph.maxflow(source)
        return 0


class DinicsSolver(AugmentingPathSolver):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.levels: dict[Node, int] = {}

    def gen_levels(self, source: Node, sink: Node):
        """Creates a layered graph/ admissible graph using breadth first search
        Variables:
            delta: the level of the sink
            lid: the level id
        """

        self.levels = {node: 0 for node in self.graph}
        self.levels[source] = 1

        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == sink:
                break
            for neighbor in self.graph.adjacency(node):
                if (
                    self.graph.residual_capacity(node, neighbor) > 0
                    and self.levels[neighbor] == 0
                ):
                    self.levels[neighbor] = self.levels[node] + 1
                    queue.append(neighbor)
        return self.levels[sink] != 0

    def gen_blocking_flow(
        self, current: Node, sink: Node, pushed: int, visited: dict[Node, set[Node]]
    ) -> int:
        if pushed == 0 or current == sink:
            return pushed
        for neighbor in self.graph.adjacency(current):
            if (
                neighbor not in visited[current]
                and self.levels[neighbor] == self.levels[current] + 1
                and self.graph.residual_capacity(current, neighbor) > 0
            ):
                visited[current].add(neighbor)
                blocking_flow = self.gen_blocking_flow(
                    neighbor,
                    sink,
                    min(pushed, self.graph.residual_capacity(current, neighbor)),
                    visited,
                )
                if blocking_flow > 0:
                    self.graph[current][neighbor].flow += blocking_flow
                    self.graph[neighbor][current].flow -= blocking_flow
                    return blocking_flow
        return 0

    def solve(self, source: Node, sink: Node):
        assert source != sink

        if self.has_path(source, sink):
            max_flow = 0
            while self.gen_levels(source, sink):
                visited: dict[Node, set[Node]] = defaultdict(set)
                while (
                    blocking_flow := self.gen_blocking_flow(source, sink, INF, visited)
                ) != 0:
                    max_flow += blocking_flow
            return max_flow
        return 0


class PushRelabelSolver(MaxFlowSolver, ABC):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.excess: dict[Node, int] = defaultdict(int)
        self.height: dict[Node, float] = {}

    def __eq__(self, other):
        return (
            isinstance(other, PushRelabelSolver)
            and self.excess == other.excess
            and self.height == other.height
            and super().__eq__(other)
        )

    def initialize_pre_flow(self, source: Node, sink: Node) -> None:
        assert source in self.graph and sink in self.graph
        # Initialize heights as the shortest distance from the sink to every node except the source
        # We perform bfs on the original graph, not the residual one

        self.height = {node: 0 for node in self.graph}
        self.height[source] = self.graph.n_nodes - 1

        # saturate all the edges coming out of the source
        for v in self.graph.adjacency(source):
            self.excess[v] = self.graph[source][v].flow = self.graph[source][v].cap
            self.excess[source] -= self.graph[source][v].cap

        for u, v, attrs in tuple(self.graph.edges_with_attrs):
            if attrs.reversed:
                continue
            self.graph.insert_edge(
                v,
                u,
                self.graph[u][v].cap,
                self.graph.residual_capacity(u, v),
                reversed=True,
            )

        # Create the residual graph of g, Q contains the nodes with positive excesses
        # Initially, those are just the edges outgoing from the source

    def push(self, u: Node, v: Node):
        """Push(u). If ∃v with admissible arc (u, v) ∈ E_f , then send flow δ := min(cf (uv), ef (u))
        from u to v. Note that this causes excess ef (u) to fall by δ, and excess ef (v) to increase
            by δ. If δ = cf (uv), this is called a saturating push, else it is a non-saturating push.
        """
        # assert self.excess[u] > 0 and self.height[u] == self.height[v] + 1

        delta = min(self.graph.residual_capacity(u, v), self.excess[u])
        self.graph[u][v].flow += delta
        self.graph[v][u].flow -= delta
        self.excess[u] -= delta
        self.excess[v] += delta

    def relabel(self, u: Node):
        valid = [
            v for v in self.graph.adjacency(u) if self.graph.residual_capacity(u, v) > 0
        ]
        # assert len(valid) != 0
        # assert self.excess[u] > 0
        # assert all(self.height[u] <= self.height[v] for v in valid)
        self.height[u] = min(self.height[v] for v in valid) + 1


class RelabelToFrontSolver(PushRelabelSolver):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.seen: dict[Node, int] = defaultdict(int)

    def discharge(self, u: Node):
        neighbors_u = tuple(self.graph[u].keys())
        while self.excess[u] > 0:
            if self.seen[u] < self.graph.num_neighbors(u):
                v = neighbors_u[self.seen[u]]
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
        assert source in self.graph and sink in self.graph

        self.graph.set_flows(0)

        self.initialize_pre_flow(source, sink)

        order = [
            node
            for node in sorted(
                self.graph.nodes, key=lambda node: self.height[node], reverse=True
            )
            if node not in (source, sink)
        ]
        # start from the higher nodes because heights correspond to BFS levels from the sink

        pos, n = 0, len(order)

        while pos < n:
            u = order[pos]
            old_height = self.height[u]
            self.discharge(u)
            if self.height[u] > old_height:
                order.insert(0, order.pop(pos))  # move to front
                pos = 0
            else:
                pos += 1

        return sum(self.graph[source][v].flow for v in self.graph.adjacency(source))


class FifoPushRelabelSolver(PushRelabelSolver):
    def solve(self, source, sink):
        """FIFO Push/Relabel
        Heuristics used:
            1. Choosing the highest vertex
            2. Initializing the heights to shortest v-t paths
        Invariants:
            1. h(s) = n at all times (where n = |V|);
            2. h(t) = 0;
            3. for every edge (v, w) of the current residual network (with positive residual capacity),
                h(v) ≤ h(w) + 1.
        excess(u) = flow(into) - flow(out)
        residual_capacity(u, v) = capacity(u, v) - flow(u, v)

        """
        self.graph.set_flows(0)

        self.initialize_pre_flow(source, sink)

        Q: list[tuple[float, Node]] = []
        queued: set[Node] = set()

        for u in self.graph.adjacency(source):
            queued.add(u)
            heappush(Q, (-self.height[u], u))
        while Q:
            # highest active node, has the lowest -(height)
            u = heappop(Q)[1]
            queued.discard(u)
            if u == sink:
                continue
            for v in self.graph.adjacency(u):
                if self.excess[u] == 0:
                    break
                if (
                    self.height[u] == self.height[v] + 1
                    and (self.graph[u][v].cap - self.graph[u][v].flow) > 0
                ):
                    self.push(u, v)
                    if v not in queued and v not in (source, sink):
                        heappush(Q, (-self.height[v], v))
                        queued.add(v)
            if self.excess[u] > 0:
                self.relabel(u)
                heappush(Q, (-self.height[u], u))
                queued.add(u)

        return sum(self.graph[source][v].flow for v in self.graph.adjacency(source))


def get_default_solver():
    return FifoPushRelabelSolver
