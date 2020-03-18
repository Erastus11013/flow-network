from edmonds_karp import FlowNetwork, defaultdict


class CapacityScaler(FlowNetwork):
    def __init__(self):
        super().__init__()

        self.U = -self.INF
        self.discovered = defaultdict(lambda: False)
        self.pred = defaultdict(lambda: None)

    def insert_edges_from_iterable(self, edges):
        for edge in edges:
            self.insert_edge(edge)
            self.U = max(self.U, self.nodes[edge[0]][edge[1]].cap)

    def augment_paths(self, source, sink, delta):
        """"""
        # mark all nodes as unvisited
        while True:
            self.mark_as_unvisited()
            S = [source]
            gf = 0
            while S:
                u = S.pop()
                if u == sink:
                    cf, _ = self.update_network(self.pred, source, sink)
                    gf = max(gf, cf)
                    continue
                if self.discovered[u]:
                    continue
                self.discovered[u] = True
                for v in self.nodes[u]:
                    if (self.nodes[u][v].cap - self.nodes[u][v].flow) >= delta and not self.discovered[v]:
                        self.pred[v] = u
                        S.append(v)
            if not gf:
                break

    def find_max_flow(self, source, sink):
        self.set_flows(0)
        self.build_residual_graph()

        delta = 1 << (self.U.bit_length() - 1)
        while delta > 0:
            self.augment_paths(source, sink, delta)
            delta >>= 1
        return self.maxflow(source)
