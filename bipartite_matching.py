from dinitz import layered_graph
from typing import List
from collections import deque
from math import inf


class bipartite_graph(layered_graph):
    def __init__(self):
        layered_graph.__init__(self)

    def is_bipartite(self, source=None):
        """ Assumes the graph is connected, presumably from the calling of self.maximum_flow() which adds
        a supersource and a supersink. Uses the 2-coloring to check for bipartiteness
        Returns true is the graph is bipartite, False otherwise"""

        if self.loop_exists():  # checks whether self-loop exists
            return False
        if source is None: source = self.supersource
        if source not in self.nodes:
            raise ValueError("Call multiple maxflows to connect the graph")
        WHITE, BLACK, RED = 2, 0, 1
        color = {node: WHITE for node in self.nodes} # UNSEEN
        source, item = self.nodes.popitem() # chooses some key from the dictionary as the source
        self.nodes[source] = item

        color[source] = BLACK

        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.nodes[u]:
                if color[v] == WHITE:
                    # if two adjacent nodes have the same color
                    color[v] = int(not color[u]) # opposite color
                    q.append(v)
                else: # node exists
                    if color[v] == color[u]:
                        return False
        return True

    def get_minimal_vertex_cover(self):
        """Return a set of edges"""
        if len(self.path) == 0:
            raise ValueError("run edmond_karps method with the store_path option")
        else:
            return self.path



def test_bipartite_matching(L, R, E):
    """Returns a maximum cardinality bipartite matching for unweighted edges
    Uses the ford-fulkerson method"""
    print("Using edmonds-karp: ")
    g = bipartite_graph()
    g.add_nodes(L)
    g.add_nodes(R)
    g.add_edges(E)
    g.set_caps(1)
    g.multiple_max_flow(L, R, cap=1)
    H = g.is_bipartite()
    print(H)
    g.edmond_karp()
    print("Using dinitz algorithm")
    max_flow = g.dinitz_algorithm()
    print(max_flow)


if __name__ == '__main__':
    people = ['a', 'b', 'c', 'd', 'e']
    books = ['b1', 'b2', 'b3', 'b4', 'b5']
    edges = [('a', 'b2'), ('a', 'b3'), ('b', 'b2'), ('b', 'b3'), ('b', 'b4'), ('c', 'b1'), ('c', 'b2'),
             ('c', 'b3'), ('c', 'b5'), ('d', 'b3'), ('e', 'b3'), ('e', 'b4'), ('e', 'b5')]
    test_bipartite_matching(people, books, edges)

