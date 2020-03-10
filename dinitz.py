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

    def _delete_saturated_edges(self, path):
        """Deletes saturated edges from the layered/level graph L."""
        for lid, edge in enumerate(path):
            src, dest, _, _ = edge
            if self.residual_capacity(src, dest) == 0:  # edge is saturated
                self.layers[src].remove(dest)  # delete the edge
        return True

    def saturate_one_path(self, source, sink, print_path=False):
        """Traverses the graph using a modified non-recursive DFS to find a find a flow to saturate
            one path in the layered graph.
            If no s -> t path exists, this function won't be called delta is the level of the sink.
            The algorithm runs in O(|V||E|) time.

            prodedure ModifiedDFS:
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
                            self._delete_saturated_edges(path)
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


def test_dinitz(nodes, edges):
    g1 = LayeredGraph()
    g1.add_nodes(nodes)
    g1.add_edges(edges)

    maxflow = g1.dinitz_algorithm('s', 't')
    del g1
    return maxflow


def test_edmonds(nodes, edges):
    g1 = FlowNetwork()
    g1.add_nodes(nodes)
    g1.add_edges(edges)
    maxflow = g1.edmond_karp('s', 't')
    del g1
    return maxflow


# s, t, a, b, c, d, e, f, g, h, i = 's', 't', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'
# n = [s, t, a, b, c, d, e, f, g, h, i]
# e = [(s, a, 5), (s, d, 10), (s, g, 15), (a, b, 10), (b, c, 10), (b, e, 25), (c, t, 5), (d, a, 15), (d, e, 20),
#      (e, f, 30), (e, g, 5),
#      (f, t, 15), (f, b, 15), (f, i, 15), (g, h, 25), (h, i, 10), (h, f, 20), (i, t, 10)]

s, v1, v2, v3, v4, v5, t = 's', 'v1', 'v2', 'v3', 'v4', 'v5', 't'
n1 = [s, v1, v2, v3, v4, t]
e1 = [(s, v1, 16), (s, v2, 13), (v1, v3, 12), (v2, v1, 4), (v2, v4, 14), (v3, v2, 9), (v3, t, 20),
      (v4, v3, 7), (v4, t, 4)]

if __name__ == '__main__':
    from time import perf_counter
    t1 = perf_counter()
    num_iters = 100
    for i in range(num_iters):
        test_dinitz(n1, e1)
    print(f"Dinitz ran in {perf_counter() - t1:.2f} seconds:")

    t1 = perf_counter()
    for i in range(num_iters):
        test_edmonds(n1, e1)
    print(f"Edmonds ran in {perf_counter() - t1:.2f} seconds:")
