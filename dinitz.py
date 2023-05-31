from collections import deque, defaultdict

from core import Node
from edmonds_karp import FlowNetwork


class LayeredGraph(FlowNetwork):
    """For implementation of dinitz's algorithm"""

    def __init__(self):
        FlowNetwork.__init__(self)

    def gen_levels(self, source: Node, sink: Node):
        """Creates a layered graph/ admissible graph using breadth first search
        Variables:
            delta: the level of the sink
            lid: the level id
        """

        queue = deque([source])
        levels = {node: 0 for node in self}
        levels[source] = 1

        while queue:
            u = queue.popleft()
            if u == sink:
                break
            for v in self.neighbors(u):
                if (self[u][v].cap - self[u][v].flow) > 0 and levels[v] == 0:
                    levels[v] = levels[u] + 1
                    queue.append(v)

        return levels

    def gen_blocking_flow(self, u, pushed, sink, levels, visited):
        if pushed == 0 or u == sink:
            return pushed
        for v in self.neighbors(u):
            if (
                v not in visited[u]
                and levels[v] == levels[u] + 1
                and self[u][v].cap - self[u][v].flow > 0
            ):
                visited[u].add(v)
                cf = self.gen_blocking_flow(
                    v,
                    min(pushed, self[u][v].cap - self[u][v].flow),
                    sink,
                    levels,
                    visited,
                )
                if cf > 0:
                    self[u][v].flow += cf
                    self[v][u].flow -= cf
                    return cf
        return 0

    def dinitz_algorithm(self, source: Node, sink: Node):
        assert source != sink
        self.set_flows(0)  # step 1
        self.init_reversed_edges()  # initialization
        max_flow = 0

        while True:
            levels = self.gen_levels(source, sink)
            visited = defaultdict(set)
            if levels[sink] == 0:
                break
            while (
                blocking_flow := self.gen_blocking_flow(
                    source, self.INF, sink, levels, visited
                )
            ) != 0:
                max_flow += blocking_flow
        return max_flow
