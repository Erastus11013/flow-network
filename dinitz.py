from edmonds_karp import FlowNetwork
from collections import defaultdict


class LayeredGraph(FlowNetwork):
    """For implementation of dinitz's algorithm"""

    __slots__ = ("visited", "layers")

    def __init__(self):
        FlowNetwork.__init__(self)
        self.layers = defaultdict(set)  # set of layers

    def create_layered_graph(self, source, sink):
        """Creates a layered graph/ admissible graph using breadth first search
            Variables:
                delta: the level of the sink
                lid: the level id
        """
        visited = {source}

        lid, frontier, next_layer = 0, [source], []
        sink_reached = False

        while frontier:
            for u in frontier:
                self.layers[u] = set()
                for v in self[u]:
                    if (self[u][v].cap - self[u][v].flow) > 0:
                        if v not in visited:
                            if v == sink:
                                sink_reached = True
                            visited.add(v)
                            next_layer.append(v)
                            self.layers[u].add(v)
            frontier = next_layer
            next_layer = []
        return sink_reached

    def del_saturated_edges(self, cf, path):
        """Deletes saturated edges from the layered/level graph L."""
        for lid, edge in enumerate(path):
            src, dest, cap = edge
            if cap == cf:  # edge is saturated
                self.layers[src].remove(dest)  # delete the edge
        return True

    def find_blocking_flow(self, source, sink):
        """ Finds a blocking flow of the layered graph by saturating one path at time.
            If no s -> t path exists, this function won't be called delta is the level of the sink.
            The algorithm runs in O(|V||E|) time.
            procedure ModifiedDFS:
                1 advance(v, w): move from v to w for (v, w) ∈ L,
                2 retreat(u, v): if (v, w) ∄ L∀w, then delete (u, v) from L
                3 augment: if v = t, augment f along the minimum residual flow on the
                found s-t path and delete saturated edges.
            Variables:
             S: a stack to simulate DFS of the graph.
             pred: a dictionary where pred[u] = v tells that u is v's predecessor."""

        while True:  # blocking flow exists
            S, pred = [source], {}  # O(1)
            cf = 0
            while S:
                u = S.pop()
                # advance if vertex u has outgoing edges
                if len(self.layers[u]) > 0:
                    for v in self.layers[u]:  # for each vertex v in neighbors_L(u)
                        if v == sink:  # augment the path by the lowest residual capacity
                            pred[v] = u
                            cf, path = self.update_network(pred, source, sink)  # augment path
                            # delete saturated edges
                            self.del_saturated_edges(cf, path)
                            # start the dfs from the source again
                            S = {}
                            break
                        else:
                            S.append(v)  # push to stack
                            pred[v] = u
                # retreat
                else:
                    # if u is the source, blocking flow has been attained
                    # delete this edge from the layered graph
                    self.layers.pop(u)
            if not cf:
                break

    def dinitz_algorithm(self, source=None, sink=None):
        """
        procedure Dinic:
            1 f(v,w) ← 0 ∀(v,w) ∈ E
            2 Create residual graph Gf
            3 Create a layered graph Lf using BFS...
            4 while Gf has an s-t path do
            5      Find a blocking flow f in Gf by multiple dfs runs
            6      Augment by f
            5 end while
        """

        self.set_flows(0)  # step 1
        self.build_residual_graph()  # initialization
        sink_reachable = self.create_layered_graph(source, sink)  # initialization

        while sink_reachable:
            self.find_blocking_flow(source, sink)
            sink_reachable = self.create_layered_graph(source, sink)
        return self.maxflow(source)
