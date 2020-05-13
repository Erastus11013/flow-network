# author: Erastus Murungi
from graph_base import *


class FlowNetwork(DiGraph):
    supersource = '@'
    supersink = '#'

    __slots__ = ("discovered", "pred")

    def __init__(self):
        DiGraph.__init__(self)

        self.discovered = defaultdict(lambda: 0)
        self.pred = defaultdict(lambda: None)

    def insert_edge(self, edge) -> None:
        super().insert_edge(edge)
        self.check_capacity_constraint(edge[0], edge[1])

    def remove_anti_parallel_edges(self):
        ap_edges = []
        for u, v in self.edges:
            if u in self[v]:
                ap_edges.append((u, v))
        for u, v in ap_edges:
            v_prime = str(u) + str(v)  # create new label by concatenating original labels
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
        self[FlowNetwork.supersource] = {source: EdgeInfo(cap) for source in sources}  # flow is 0
        for sink in sinks:
            self[sink][FlowNetwork.supersink].cap = cap
        return True

    def build_residual_graph(self):
        """"""
        edges = tuple(self.edges)
        for u, v in edges:
            c = self[u][v].cap
            self[v][u] = EdgeInfo(c, c, 1)

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

    def augmenting_path(self, pred, source, sink):
        """Returns an iterator to help in updating the original graph G
        Returns a list of tuples in the format (source, dest, residual_capacity, flipped)
        (source, dest): an edge in G_f, it might be reversed or not
        flipped: tells whether the edge had been reversed in the original graph G"""

        path = []
        curr = sink
        while curr != source:
            p = pred[curr]
            path.append((p, curr, self[p][curr].cap - self[p][curr].flow))
            curr = p
        return reversed(path)

    def maxflow(self, source):
        val_f = 0
        for v in self[source]:
            val_f += self[source][v].flow
        return val_f

    def check_capacity_constraint(self, src, dst):
        """sanity check"""
        if self[src][dst].flow > self[src][dst].cap:
            raise ValueError("capacity cannot be less than the flow")

    def set_flows(self, val):
        for u in self:
            for v in self[u]:
                self[u][v].flow = val

    def set_caps(self, val):
        for u in self:
            for v in self[u]:
                self[u][v].cap = val

    def _update(self, path, cf):
        for arg in path:
            src, dst, flow = arg
            self[dst][src].flow -= cf
            self[src][dst].flow += cf

    def mark_as_unvisited(self):
        for u in self.discovered:
            self.discovered[u] = 0
        for u in self.pred:
            self.pred[u] = None

    def bfs_residual_graph(self, source):
        self.mark_as_unvisited()
        q = deque([source])
        while q:
            u = q.pop()
            for v in self[u]:
                if self[u][v].cap - self[u][v].flow > 0:
                    if not self.discovered[v]:
                        self.discovered[v] = self.discovered[u] + 1
                        self.pred[v] = u
                        q.appendleft(v)
        return self.pred

    def update_network(self, pred, source, sink, print_path=False) -> Tuple[float, List]:
        """uses the predecessor dictionary to determine a path
        the path is a list of tuples where each tuple is the format"""

        path = list(self.augmenting_path(pred, source,
                                         sink))
        if print_path:
            self.print_path(pred, source, sink)
        cf = min(edge[2] for edge in path)  # tup[2] contains the flow
        self._update(path, cf)
        return cf, path

    def edmonds_karp(self, source=None, sink=None, print_path=False):
        """ Edmonds Karp implementation of the Ford Fulkerson method
            Notice:
            if graph may have anti-parallel edges:
                add this line: self.remove_anti_parallel_edges()  # u -> v ==> u -> v'; v' -> v
            if graph may have self-loops:
                add this line: self.remove_self_loops()
        """

        self.set_flows(0)  # f[u,v] = 0
        self.build_residual_graph()
        pred = self.bfs_residual_graph(source)  # run bfs once

        # while augmenting path exists
        while pred[sink] is not None:
            self.update_network(pred, source, sink, print_path)
            pred = self.bfs_residual_graph(source)
        return self.maxflow(source)
