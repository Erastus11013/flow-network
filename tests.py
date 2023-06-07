import pytest

from applications import maximum_bipartite_matching
from core import Digraph, FlowNetwork, Node
from solvers import (
    CapacityScalingSolver,
    DinicsSolver,
    EdmondsKarpSolver,
    FifoPushRelabelSolver,
    RelabelToFrontSolver,
)
from utils import gen_random_network


def read_flow_net_from_file(filename: str) -> tuple[FlowNetwork, Node, Node, int]:
    flow_network = FlowNetwork()
    with open(filename + ".sol", "r") as f:
        contents = f.readlines()
        lineno = 0
        while not contents[lineno].startswith("s"):
            lineno += 1
        max_flow = int(contents[lineno].split()[1])

    with open(filename + ".max", "r") as f:
        contents = f.readlines()
        lineno = 0
        while not contents[lineno].startswith("p"):
            lineno += 1
        line = contents[lineno].split()
        num_nodes, num_edges = int(line[2]), int(line[3])
        while contents[lineno][0] != "n":
            lineno += 1
        while contents[lineno][0] == "n":
            line = contents[lineno].split()
            if line[2] == "s":
                source = Node(line[1])
            elif line[2] == "t":
                sink = Node(line[1])
            lineno += 1

        for i in range(lineno, len(contents)):
            line = contents[i].split()
            if len(line) == 0:
                continue
            if line[0] == "a":
                flow_network.insert_edge(Node(line[1]), Node(line[2]), int(line[3]))
            else:
                continue
    del contents
    assert flow_network.n_nodes == num_nodes
    assert flow_network.n_edges() == num_edges
    return flow_network, source, sink, max_flow


@pytest.mark.parametrize(
    "num_nodes, max_outgoing, max_cap, num_iters",
    [(400, 200, 500, 1), (10, 10, 500, 1000)],
)
def test_all_matching_using_random_flow_networks(
    num_nodes: int, max_outgoing: int, max_cap: int, num_iters: int
):
    for _ in range(num_iters):
        flow_network, source, sink = gen_random_network(
            num_nodes, max_outgoing, max_cap
        )
        results = []
        for solver in (
            FifoPushRelabelSolver(flow_network),
            RelabelToFrontSolver(flow_network),
            EdmondsKarpSolver(flow_network),
            DinicsSolver(flow_network),
            CapacityScalingSolver(flow_network),
        ):
            maxflow = solver.solve(source, sink)
            results.append(maxflow)
        assert results[0] == results[1] == results[2] == results[3] == results[4]


def test_is_bipartite():
    digraph = Digraph()
    digraph.insert_edges_from_iterable(
        [(Node(1), Node(3), Node(0)), (Node(1), Node(2), 0), (Node(2), Node(4), 0)]
    )
    assert digraph.is_bipartite()


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
    matching = maximum_bipartite_matching(
        list(map(Node, people)), list(map(Node, books)), edges, 1
    )
    assert len(matching) == 5


# def _test_LB07_bunny_sml():
#     graph, source, sink, expected_maxflow = read_flow_net_from_file("testdata/LB07-bunny-sml/LB07-bunny-sml")
#     solver = FifoPushRelabelSolver(graph)
#     maxflow = solver.solve(source, sink)
#     assert maxflow == expected_maxflow
#
# if __name__ == '__main__':
#     _test_LB07_bunny_sml()
