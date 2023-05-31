from pprint import pprint
from sys import maxsize

from push_relabel import *


def isbipartite(g: Digraph) -> bool:
    """A bipartite graph (or bigraph) is a graph whose vertices can be divided into two disjoint
        and independent sets U and V such that every edge connects a vertex in U to one in V.
        Vertex sets U and V are usually called the parts of the graph.
        Equivalently, a bipartite graph is a graph that does not contain any odd-length cycles.

    Let G be a graph. Then G is 2-colorable if and only if G is bipartite.
    source: https://cp-algorithms.com/graph/bipartite-check.html
    """

    color: dict[Node, int] = defaultdict(lambda: -1)  # Literal[-1, 0, 1]
    Q: deque[Node] = deque()
    is_bipartite = True

    for source in g.nodes():
        if color[source] == -1:
            Q.appendleft(source)
            color[source] = 0
            while Q:
                v = Q.pop()
                for u, _ in g.adjacency(v):
                    if color[u] == -1:
                        color[u] = color[v] ^ 1
                        Q.appendleft(u)
                    else:
                        is_bipartite &= color[u] != color[v]
    return is_bipartite


def match(U, V, E, max_cap):
    """Takes at input the left and right edges
    Given a bipartite graph G = (A ∪ B, E), find an S ⊆ A × B that is
    a matching and is as large as possible."""

    g = Digraph()
    if len(E[0]) == 2:
        E = tuple(map(lambda arc: arc + (1,), E))
    g.insert_edges_from_iterable(E)
    if not isbipartite(g):
        raise ValueError("The graph must be bipartite for maximum bipartite matching.")
    # add supersink and supersource
    supersource, supersink = Node(-maxsize), Node(maxsize)
    for source in U:
        g.insert_edge(supersource, source, max_cap)  # flow is 0
    for sink in V:
        g.insert_edge(sink, supersink, max_cap)
    maxflow = g.fifo_push_relabel(supersource, supersink)
    S = [(u, v, g[u][v].flow) for (u, v, _) in E]
    return S, maxflow


def test_is_bipartite():
    g = Digraph()
    g.insert_edges_from_iterable([(1, 3, 0), (1, 2, 0), (2, 4, 0)])
    print(isbipartite(g))


def test_bipartite_matching():
    people = ["p1", "p2", "p3", "p4", "p5"]
    books = ["b1", "b2", "b3", "b4", "b5"]
    edges = [
        ("p1", "b2"),
        ("p1", "b3"),
        ("p2", "b2"),
        ("p2", "b3"),
        ("p2", "b4"),
        ("p3", "b1"),
        ("p3", "b2"),
        ("p3", "b3"),
        ("p3", "b5"),
        ("p4", "b3"),
        ("p5", "b3"),
        ("p5", "b4"),
        ("p5", "b5"),
    ]

    print("using fifo push-relabel... ")
    g = Digraph()
    g.insert_edges_from_iterable(map(lambda arc: arc + (1,), edges))
    S, maxflow = match(people, books, edges, 1)
    pprint(S)
    pprint(maxflow)


if __name__ == "__main__":
    test_bipartite_matching()
