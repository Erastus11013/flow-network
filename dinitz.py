from graph import FlowNetwork
from collections import defaultdict


class LayeredGraph(FlowNetwork):
    """For implementation of dinitz's algorithm"""

    def __init__(self):
        FlowNetwork.__init__(self)
        self.layers = {}  # set of layers

    def create_layered_graph(self, source, sink):
        """Creates a layered graph/ admissible graph using breadth first search
            Variables:
                delta: the level of the sink
                lid: the level id
        """
        visited = defaultdict(lambda: False)
        visited[source] = True
        lid, frontier = 0, [source]
        delta = None

        while frontier:
            lid += 1
            next_layer = []
            for u in frontier:
                self.layers[u] = set()
                for v in self.neighbors_in_residual_graph(u):
                    if not visited[v]:
                        if v == sink:
                            delta = lid
                        visited[v] = True
                        next_layer.append(v)
                        self.layers[u].add(v)
            frontier = next_layer
        return delta

    def _delete_saturated_edges(self, cf, path):
        """Deletes saturated edges from the layered/level graph L."""
        for lid, edge in enumerate(path):
            src, dest, cap, _ = edge
            if cap == cf:  # edge is saturated
                self.layers[src].remove(dest)  # delete the edge
        return True

    def saturate_one_path(self, source, sink, print_path=True):
        """Traverses the graph using a modified non-recursive DFS to find a find a flow to saturate
            one path in the layered graph.
            If no s -> t path exists, this function won't be called delta is the level of the sink.
            The algorithm runs in O(|V||E|) time.

            procedure ModifiedDFS:
                1 advance(v, w): move from v to w for (v, w) ∈ L,
                2 retreat(u, v): if (v, w) ∄ L∀w, then delete (u, v) from L
                3 augment: if v = t, augment f along the minimum residual flow on the
                found s-t path and delete saturated edges.
            Variables:
             S: a stack to simulate DFS of the graph.
             pred: a dictionary where pred[u] = v tells that u is v's predecessor.
        """

        S, pred = [source], {}  # O(1)
        cf = 0

        while S:
            u = S.pop()
            if u in self.layers:
                if self.layers[u]:  # advance if vertex u has outgoing edges
                    for v in self.layers[u]:  # for each vertex v in neighbors_L(u)
                        if v == sink:  # augment the path by the lowest residual capacity
                            pred[v] = u
                            cf, path = self._augment(pred, source, sink, print_path)  # augment path
                            # delete saturated edges
                            self._delete_saturated_edges(cf, path)
                            # start the dfs from the source again
                            S.clear()
                            break  # clear the stack and stop
                        else:
                            S.append(v)  # push to stack
                            pred[v] = u

                else:  # retreat
                    if u == source:  # if u is the source, blocking flow has been attained
                        S.clear()
                    else:
                        # delete last edge from the layered graph
                        self.layers.pop(u)
        return cf

    def find_blocking_flow(self, source, sink):
        """ Finds a blocking flow of the layered graph by saturating one path at time. """
        bf = 0
        while len(self.layers[source]):  # blocking flow exists
            cf = self.saturate_one_path(source, sink, print_path=False)
            if cf:
                bf += cf
            else:
                return bf
        return bf

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
        if source is None:
            source = self.supersource
        if sink is None:
            sink = self.supersink
        if source not in self.nodes:
            return None

        total_flow = 0
        self.set_flows(0)  # step 1
        self.create_residual_graph()  # initialization
        delta = self.create_layered_graph(source, sink)  # initialization

        if delta is None:  # if the sink is unreachable, max flow is already attained
            return self.get_max_flow(source)

        while delta is not None:
            blocking_flow = self.find_blocking_flow(source, sink)
            if blocking_flow == 0:  # max flow has been reached
                return self.get_max_flow(source)

            total_flow += blocking_flow
            self.create_residual_graph()
            delta = self.create_layered_graph(source, sink)
            assert total_flow == self.get_max_flow(source)

        return self.get_max_flow(source)
