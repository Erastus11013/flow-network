from ford_fulkerson import flow_network
from collections import deque
from math import inf


class layered_graph(flow_network):
    """For implementation of dinitz's algorithm"""
    MAX_LEVELS = 13

    def __init__(self):
        flow_network.__init__(self)
        self.layers: List[set] = [0] * layered_graph.MAX_LEVELS  # set of layers

    def create_layered_graph(self, source, sink):
        """Creates a layered graph/ admissible graph using breadth first search

        """
        visited = {node: False for node in self.nodes}
        visited[source] = True
        lid, layer, frontier = 0, set(), [source]
        self.layers[lid] = set(source)
        delta = None

        while frontier:
            lid += 1
            next_layer = []
            for u in frontier:
                for v in self.neighbors_in_residual_graph(u):
                    if not visited[v]:
                        if v == sink: delta = lid
                        layer.add(v)
                        visited[v] = True
                        next_layer.append(v)
            if layer: self.layers[lid] = layer
            layer = set()
            frontier = next_layer
        return delta

    def __delete_from_layer_graph(self, pred, sink):
        """Assumes the pred dict is ordered"""
        j = 1
        for k in pred:
            if k != sink:
                self.layers[j].remove(k)
                j += 1
            else:
                return True

    def saturate_one_path(self, source, start, sink, delta, print_path=True):
        """Uses Iterative DFS
          After reaching the end, destroy the whole path from the beginning. How do we do that?
            Multiple dfs searches
            If no s -> t path exists, this function won't be called
        O(V + E)"""
        visited, stack, pred = set(), [start], {start: source}  # O(1)
        lid = 2

        while stack:   # O(V)
            u = stack.pop()
            if lid > delta:  # the sink is unreachable
                return 0
            if u not in visited:
                visited.add(u)
                for v in self.neighbors_in_residual_graph(u):
                    if v not in visited and v in self.layers[lid]:
                        if v == sink: # that's it
                            pred[v] = u
                            cf = self._augment(pred, source, sink, print_path)
                            self.__delete_from_layer_graph(pred, sink)
                            return cf
                        pred[v] = u
                        stack.append(v)
                        lid += 1
        return 0

    def find_blocking_flow(self, source, sink, delta):
        bf = 0
        for u in self.neighbors_in_residual_graph(source):
            bf += self.saturate_one_path(source, u, sink, delta, True)
        return bf

    def sink_is_reachable(self, delta, sink):
        return sink in self.layers[delta]

    def dinitz_algorithm(self, source=None, sink=None):
        """
        1: f(v,w) ← 0 ∀(v,w) ∈ E
        2: Create residual graph Gf
        3: Create a layered graph Lf using BFS...
        4: while Gf has an s-t path do
        5:      Find a blocking flow f in Gf by multiple dfs runs
        6:      Augment by f
        5: end while
        """
        if source is None: source = self.supersource
        if sink is None: sink = self.supersink
        if source not in self.nodes:
            return None

        total_flow = 0
        self.set_flows(0)  # step 1   # works as expected
        self.create_residual_graph() # initialization
        delta = self.create_layered_graph(source, sink)   # initialization
        if delta is None: return self.get_max_flow(source)
        while delta is not None:
            blocking_flow = self.find_blocking_flow(source, sink, delta)
            total_flow += blocking_flow
            self.create_residual_graph()
            delta = self.create_layered_graph(source, sink)
        return self.get_max_flow(source)

def test_dinitz(nodes, edges):
    g1 = layered_graph()
    g1.add_nodes(nodes)
    g1.add_edges(edges)
    v1 = g1.dinitz_algorithm('s', 't')
    print(v1)
    v2 = g1.edmond_karp('s', 't', 't')
    print(v2)
    return v1


s, t, a, b, c, d, e, f, g, h, i = 's', 't', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'
n = [s, t, a, b, c, d, e, f, g, h, i]
e = [(s, a, 5), (s, d, 10), (s, g, 15), (a, b, 10), (b,c, 10), (b,e, 25), (c, t, 5), (d, a, 15), (d, e, 20), (e, f, 30), (e, g, 5),
     (f, t, 15), (f, b, 15), (f, i, 15), (g, h, 25), (h, i, 10), (h, f, 20), (i, t, 10)]

if __name__ == '__main__':
    test_dinitz(n ,e)

