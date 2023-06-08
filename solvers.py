from abc import ABC, abstractmethod
from collections import defaultdict, deque

from core import INF, FlowNetwork, Node, Predecessors
from utils import MinHeapSet


class MaxFlowSolver(ABC):
    def __init__(self, graph: FlowNetwork):
        self.original_graph = graph
        # create residual graph
        self.graph = graph.copy()
        self.graph.reset_flows()
        self.graph.initialize_reversed_edges()

    def solve(self, source: Node, sink: Node) -> int:
        """Returns the maximum flow from source to sink in the graph."""
        if not isinstance(source, Node):
            raise TypeError("The source must be a node.")
        elif not isinstance(sink, Node):
            raise TypeError("The sink must be a node.")
        elif source == sink:
            raise ValueError("The source and sink must be different.")
        elif source not in self.graph:
            raise ValueError("The source must be in the graph.")
        elif sink not in self.graph:
            raise ValueError("The sink must be in the graph.")
        return self._solve_impl(source, sink)

    @abstractmethod
    def _solve_impl(self, source: Node, sink: Node) -> int:
        pass


class AugmentingPathSolver(MaxFlowSolver, ABC):
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
            assert pred_current is not None
            self.graph[pred_current][current].flow += bottleneck
            self.graph[current][pred_current].flow -= bottleneck
            current = pred_current


class EdmondsKarpSolver(AugmentingPathSolver):
    def _solve_impl(self, source: Node, sink: Node) -> int:
        """Edmonds Karp implementation of the Ford Fulkerson method
        Notice:
        if graph may have some antiparallel edges:
            add this line: self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        if graph may have self-loops:
            add this line: self.remove_self_loops()
        """

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

    def _solve_impl(self, source: Node, sink: Node) -> int:
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

    def gen_levels(self, source: Node, sink: Node) -> bool:
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
        self, current: Node, sink: Node, pushed: int, nxt: dict[Node, int]
    ) -> int:
        if pushed == 0 or current == sink:
            return pushed
        for neighbor in self.graph.ordered_neighbors_list(current)[nxt[current] :]:
            if (
                self.levels[neighbor] == self.levels[current] + 1
                and self.graph.residual_capacity(current, neighbor) > 0
            ):
                nxt[current] += 1
                blocking_flow = self.gen_blocking_flow(
                    neighbor,
                    sink,
                    min(pushed, self.graph.residual_capacity(current, neighbor)),
                    nxt,
                )
                if blocking_flow > 0:
                    self.graph[current][neighbor].flow += blocking_flow
                    self.graph[neighbor][current].flow -= blocking_flow
                    return blocking_flow
        return 0

    def _solve_impl(self, source: Node, sink: Node) -> int:
        if self.has_path(source, sink):
            max_flow = 0
            while self.gen_levels(source, sink):
                nxt: dict[Node, int] = defaultdict(int)
                while blocking_flow := self.gen_blocking_flow(source, sink, INF, nxt):
                    max_flow += blocking_flow
            return max_flow
        return 0


class PushRelabelSolver(MaxFlowSolver, ABC):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.excess: dict[Node, int] = defaultdict(int)
        self.height: dict[Node, float] = defaultdict(int)

    def initialize_pre_flow(self, source: Node, sink: Node) -> None:
        # Initialize heights as the shortest distance from the sink to every node except the source
        # We perform bfs on the original graph, not the residual one
        self.height = self.original_graph.shallow_reverse().bfs(sink)
        self.height[source] = self.graph.n_nodes - 1

        # saturate all the edges coming out of the source
        for v in self.graph.adjacency(source):
            self.excess[v] = self.graph[source][v].flow = self.graph[source][v].cap
            self.excess[source] -= self.graph[source][v].cap
            self.graph[v][source].flow = 0

    def push(self, u: Node, v: Node) -> None:
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

    def relabel(self, u: Node) -> None:
        # assert len(valid) != 0
        # assert self.excess[u] > 0
        # assert all(self.height[u] <= self.height[v] for v in valid)
        self.height[u] = 1 + min(
            self.height[v]
            for v in self.graph.adjacency(u)
            if self.graph.residual_capacity(u, v) > 0
        )


class RelabelToFrontSolver(PushRelabelSolver):
    def __init__(self, graph: FlowNetwork):
        super().__init__(graph)
        self.seen: dict[Node, int] = defaultdict(int)

    def discharge(self, node: Node):
        neighbors = self.graph.ordered_neighbors_list(node)
        while self.excess[node] > 0:
            if self.seen[node] < self.graph.num_neighbors(node):
                v = neighbors[self.seen[node]]
                if (
                    self.graph.residual_capacity(node, v) > 0
                    and self.height[node] == self.height[v] + 1
                ):
                    self.push(node, v)
                else:
                    self.seen[node] += 1
            else:
                self.relabel(node)
                self.seen[node] = 0

    def _solve_impl(self, source: Node, sink: Node) -> int:
        self.initialize_pre_flow(source, sink)
        # start from the higher nodes because heights correspond to BFS levels from the sink
        order = [
            node
            for node in sorted(
                self.graph.nodes, key=lambda node: self.height[node], reverse=True
            )
            if node != source and node != sink
        ]
        node_index, n_nodes = 0, len(order)

        while node_index < n_nodes:
            node = order[node_index]
            old_height = self.height[node]
            self.discharge(node)
            if self.height[node] > old_height:
                order.insert(0, order.pop(node_index))  # move to front
                node_index = 0
            else:
                node_index += 1

        return sum(self.graph[source][v].flow for v in self.graph.adjacency(source))


class FifoPushRelabelSolver(PushRelabelSolver):
    def _solve_impl(self, source: Node, sink: Node) -> int:
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
        self.initialize_pre_flow(source, sink)

        heap = MinHeapSet[Node](
            (-self.height[neighbor], neighbor)
            for neighbor in self.graph.adjacency(source)
        )

        while heap:
            # highest active node, has the lowest -(height)
            if (node := heap.extract()) == sink:
                continue
            for neighbor in self.graph.adjacency(node):
                if self.excess[node] == 0:
                    break
                if (
                    self.height[node] == self.height[neighbor] + 1
                    and self.graph.residual_capacity(node, neighbor) > 0
                ):
                    self.push(node, neighbor)
                    if neighbor not in heap and neighbor not in (source, sink):
                        heap.add(neighbor, -self.height[neighbor])
            if self.excess[node] > 0:
                self.relabel(node)
                heap.add(node, -self.height[node])

        return sum(
            self.graph[source][neighbor].flow
            for neighbor in self.graph.adjacency(source)
        )


def get_default_solver():
    return FifoPushRelabelSolver
