from core import *


super_source = Node(-maxsize)
super_sink = Node(maxsize)


class FlowNetwork(DiGraph):
    __slots__ = ("discovered", "pred")

    def __init__(self):
        DiGraph.__init__(self)

    def insert_edge(self, edge) -> None:
        super().insert_edge(edge)

    def remove_anti_parallel_edges(self):
        ap_edges = []
        for u, v in self.edges:
            if u in self[v]:
                ap_edges.append((u, v))
        for u, v in ap_edges:
            v_prime = str(u) + str(
                v
            )  # create new label by concatenating original labels
            cap_flow_st = self[u].pop(v)
            self[u][v_prime] = cap_flow_st
            self[v_prime][v] = cap_flow_st
        return True

    def remove_self_loops(self):
        s_loop = []
        for u in self:
            for v in self[u]:
                if v == u:
                    s_loop.append(u)
        for u in s_loop:
            self[u].pop(u)
        return True

    def multiple_max_flow(self, sources: Iterable, sinks: Iterable, cap=1):
        self[super_source] = {
            source: EdgeAttributes(cap) for source in sources
        }  # flow is 0
        for sink in sinks:
            self[sink][super_sink].cap = cap
        return True

    def init_reversed_edges(self):
        """"""
        edges = tuple(self.edges)
        for u, v in edges:
            c = self[u][v].cap
            self[v][u] = EdgeAttributes(c, c, 1)

    def maxflow(self, source):
        val_f = 0
        for v in self[source]:
            val_f += self[source][v].flow
        return val_f

    def check_constraints(self):
        sum_all = 0
        for src in self:
            for dst in self[src]:
                sum_all += self[src][dst].flow
                assert (
                    self[src][dst].flow <= self[src][dst].cap
                ), f"capacity constraint violated ({src.id}, {dst.id}, {self[src][dst].flow}, {self[src][dst].cap})"
        assert sum_all == 0, f"flow conservation constraint violated {sum_all}"

    def set_flows(self, val):
        for u in self:
            for v in self[u]:
                self[u][v].flow = val

    def _bfs(self, parent: Predecessors, source: Node, sink: Node) -> int:
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

    def edmonds_karp(self, source: Node, sink: Node) -> int:
        """Edmonds Karp implementation of the Ford Fulkerson method
        Notice:
        if graph may have anti-parallel edges:
            add this line: self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        if graph may have self-loops:
            add this line: self.remove_self_loops()
        """
        self.set_flows(0)  # f[u,v] = 0
        self.init_reversed_edges()
        parent = Predecessors()
        max_flow = 0

        while (flow := self._bfs(parent, source, sink)) != 0:
            max_flow += flow
            v = sink
            while v != source:
                u = parent[v]
                self[u][v].flow += flow
                self[v][u].flow -= flow
                v = u
        return max_flow
