from pprint import pprint

from core import Digraph, FlowNetwork, Node, super_sink, super_source
from solvers import FifoPushRelabelSolver


def match(A: list[Node], B: list[Node], E: list[tuple[Node, Node, int]], max_cap: int):
    """Takes at input the left and right edges
    Given a bipartite graph G = (A ∪ B, E), find an S ⊆ A × B that is
    a matching and is as large as possible."""

    digraph = FlowNetwork()
    digraph.insert_edges_from_iterable(E)
    if not digraph.is_bipartite():
        raise ValueError("The graph must be bipartite for maximum bipartite matching.")
    # add a super sink and a super source
    for source in A:
        digraph.insert_edge(super_source, source, max_cap)
    for sink in B:
        digraph.insert_edge(sink, super_sink, max_cap)
    solver = FifoPushRelabelSolver(digraph)
    maxflow = solver.solve(super_source, super_sink)
    matching = [(u, v) for (u, v, _) in E if solver.graph[u][v].flow == 1]
    return matching, maxflow


def test_is_bipartite():
    digraph = Digraph()
    digraph.insert_edges_from_iterable(
        [(Node(1), Node(3), Node(0)), (Node(1), Node(2), 0), (Node(2), Node(4), 0)]
    )
    print(digraph.is_bipartite())


def test_bipartite_matching():
    people = ["p1", "p2", "p3", "p4", "p5"]
    books = ["b1", "b2", "b3", "b4", "b5"]
    edges_str = [
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

    edges = list(map(lambda arc: (Node(arc[0]), Node(arc[1]), 1), edges_str))
    matching, maxflow = match(list(map(Node, people)), list(map(Node, books)), edges, 1)
    pprint(matching)
    pprint(maxflow)


if __name__ == "__main__":
    test_bipartite_matching()
