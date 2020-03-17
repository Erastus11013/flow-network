# author: Erastus Murungi
from graph_base import *


class FlowNetwork(Graph):
    supersource = 'S'
    supersink = 'T'

    def __init__(self):
        Graph.__init__(self)
        self.residual_edges = dict()
        self.path = set()

    def insert_edge(self, edge) -> None:
        Graph.insert_edge(self, edge)
        self.check_capacity_constraint(edge[0], edge[1])

    def remove_anti_parallel_edges(self):
        """Because of RuntimeError: dictionary changed size during iteration,
        we have to modify the dictionary after iteration"""
        ap_edges = []
        for u, v in self.edges:
            if u in self.nodes[v]:
                ap_edges.append((u, v))
        for u, v in ap_edges:
            v_prime = str(u) + str(v)  # create new label by concatenating original labels
            cap_flow_st = self.nodes[u].pop(v)
            self.nodes[u][v_prime] = cap_flow_st
            self.nodes[v_prime][v] = cap_flow_st
        return True

    def remove_self_loops(self):
        s_loop = []
        for u in self.nodes:
            for v in self.nodes[u]:
                if v == u:
                    s_loop.append(u)

        for u in s_loop:
            del self.nodes[u][u]
        return True

    def multiple_max_flow(self, sources: Iterable, sinks: Iterable, cap=inf):
        """Accepts multiple sinks and multiple sources"""
        for source in sources:
            if source not in self.nodes:
                raise KeyError("make sure the source, " + str(source) + " has been added to the graph")
        for sink in sinks:
            if sink not in self.nodes:
                raise KeyError("make sure the sink, " + str(sink) + " has been added to the graph")

        self.nodes[FlowNetwork.supersource] = {source: (cap, 0) for source in sources}  # flow is 0
        for sink in sinks:
            self.nodes[sink][FlowNetwork.supersink] = (cap, 0)
        self.nodes[FlowNetwork.supersink] = {}
        return True

    def create_residual_graph(self):
        """"""
        self.residual_edges = defaultdict(dict)

        for u in self.nodes:
            for v in self.neighbors(u):
                if self.residual_capacity(u, v) > 0:
                    self.residual_edges[u][v] = EdgeInfo(self.residual_capacity(u, v), 0, 0)
                    # the direction of the edge has been reversed
                if self.flow(u, v) > 0:
                    self.residual_edges[v][u] = EdgeInfo(self.flow(u, v), 0, 1)

    def parallel_bfs(self, source):
        pass

    @staticmethod
    def print_path(pred, source, sink):
        p = []
        curr = sink
        while curr is not None and curr != source:
            p.append(curr)
            curr = pred[curr]
        p.append(source)
        if curr is None:
            print("No path from source to ", sink)
        else:
            pprint('->'.join(reversed(p)))

    @staticmethod
    def augmenting_path_exists(pred, sink):
        """If the sink has no predecessors, then there is no path from source s to sink t"""
        return not (pred[sink] is None)

    def augmenting_path(self, pred, source, sink):
        """Returns an iterator to help in updating the original graph G
        Returns a list of tuples in the format (source, dest, residual_capacity, flipped)
        (source, dest): an edge in G_f, it might be reversed or not
        flipped: tells whether the edge had been reversed in the original graph G"""

        path = []
        curr = sink
        while curr != source:
            path.append((pred[curr], curr, self.residual_edges[pred[curr]][curr].cap,
                         self.residual_edges[pred[curr]][curr].weight))
            curr = pred[curr]
        return reversed(path)

    def neighbors_in_residual_graph(self, node):
        return self.residual_edges[node]

    def residual_capacity(self, src, dst):
        """cf(u,v) = c(u,v) - f(u,v)"""
        return self.capacity(src, dst) - self.flow(src, dst)

    def capacity(self, src, dst):
        """Getter method for readability"""
        return self.nodes[src][dst].cap

    def flow(self, src, dst):
        return self.nodes[src][dst].flow

    def weight(self, src, dst):
        """The bfs assumes that every edge has weight 1"""
        return 1

    def get_max_flow(self, source):
        val_f = 0
        for v in self.neighbors(source):
            val_f += self.flow(source, v)
        return val_f

    def check_capacity_constraint(self, src, dst):
        """sanity check"""
        if self.nodes[src][dst].flow > self.nodes[src][dst].cap:
            raise ValueError("capacity cannot be less than the flow")

    def set_flows(self, val):
        for u in self.nodes:
            for v in self.nodes[u]:
                self.nodes[u][v].flow = val

    def set_caps(self, val):
        for u in self.nodes:
            for v in self.nodes[u]:
                self.nodes[u][v].cap = val

    def update_network_flow(self, path, cf):
        for arg in path:
            src, dst, flow, flipped = arg
            if flipped:
                self.nodes[dst][src].flow -= cf
            else:
                self.nodes[src][dst].flow += cf

    def bfs_residual_graph(self, source):
        discovered = defaultdict(lambda: self.INF)
        pred = defaultdict(lambda: None)
        discovered[source] = 0

        q = deque([source])
        while q:
            u = q.pop()
            for v in self.neighbors_in_residual_graph(u):
                if discovered[v] == self.INF:
                    discovered[v] = discovered[u] + 1
                    pred[v] = u
                    q.appendleft(v)
        return pred

    def _augment(self, pred, source, sink, print_path=False) -> Tuple[float, List]:
        """uses the predecessor dictionary to determine a path
        the path is a list of tuples where each tuple is the format
        (source, dest, residual_capacity, is_reversed) """

        path = list(self.augmenting_path(pred, source,
                                         sink))  # it makes a difference that the reversed iterator is converted to a list
        if print_path:
            self.print_path(pred, source, sink)
        cf = min([tup[2] for tup in path])  # tup[2] contains the flow
        self.update_network_flow(path, cf)
        return cf, path

    def edmond_karp(self, source=None, sink=None, print_path=False):
        """Edmond Karp algorithm of the Ford Fulkerson method
        Track tells which graph to print path from node to track
        Can be used for bipartite matching as well"""

        # instead creating a list to store the path every time, lets create use list
        # the whole time

        if source is None:
            source = FlowNetwork.supersource
        if sink is None:
            sink = FlowNetwork.supersink
        if source not in self.nodes:  # sanity check
            return None
        self.remove_self_loops()
        total_flow = 0
        self.set_flows(0)  # f[u,v] = 0
        self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
        self.create_residual_graph()
        pred = self.bfs_residual_graph(source)  # run bfs once

        while FlowNetwork.augmenting_path_exists(pred, sink):
            cf, _ = self._augment(pred, source, sink, print_path)
            total_flow += cf
            self.create_residual_graph()
            pred = self.bfs_residual_graph(source)

        assert (self.get_max_flow(source) == total_flow)  # sanity check
        return self.get_max_flow(source)
