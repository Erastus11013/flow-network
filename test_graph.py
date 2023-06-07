import gc
from pprint import pprint

import numpy as np

from core import FlowNetwork, Node
from solvers import (
    CapacityScalingSolver,
    DinicsSolver,
    EdmondsKarpSolver,
    FifoPushRelabelSolver,
    RelabelToFrontSolver,
)

# def graph_dinitz():
#     graph = LayeredGraph()
#     return (graph,) + fill(graph)
#
#
# def graph_edmonds():
#     graph = FlowNetwork()
#     return (graph,) + fill(graph)


def fill(graph):
    with open("testdata/data2.max", "r") as f:
        # list of strings
        contents = f.readlines()
        j = 0
        while contents[j][0] != "p":
            j += 1
        L = contents[j].split()
        V, E = int(L[2]), int(L[3])
        while contents[j][0] != "n":
            j += 1
        while contents[j][0] == "n":
            L = contents[j].split()
            if L[2] == "s":
                source = int(L[1])
            elif L[2] == "t":
                sink = int(L[1])
            j += 1

        for i in range(j, len(contents)):
            L = contents[i].split()
            if len(L) == 0:
                continue
            if L[0] == "a":
                graph.add_edges((int(L[1]), int(L[2]), int(L[3])))
            else:
                continue
    del contents
    return V, E, source, sink


def gen_net(num_nodes, branching_factor, max_cap):
    num_nodes += 1
    data = []
    for u in range(1, num_nodes):
        out = set(
            np.random.randint(
                u + 1,
                num_nodes + 1,
                np.random.randint(branching_factor >> 1, branching_factor),
            )
        )
        for v in out:
            data.append((Node(u), Node(v), np.random.randint(0, max_cap)))
    return data


def fill_graph_push_relabel(nn, b, max_cap):
    e = gen_net(nn, b, max_cap)
    g = FlowNetwork()
    for edge in e:
        g.insert_edge(*edge)
    return e, g, nn, len(e), Node(1), Node(nn + 1)


def fill_test_graph():
    g = FlowNetwork()
    edges = [
        (Node(id=1, attributes={}), Node(id=3, attributes={}), 257),
        (Node(id=2, attributes={}), Node(id=3, attributes={}), 411),
        (Node(id=2, attributes={}), Node(id=4, attributes={}), 486),
        (Node(id=3, attributes={}), Node(id=4, attributes={}), 242),
    ]
    for u, v, cap in edges:
        g.insert_edge(u, v, cap)
    return edges, g, 3, 2, Node(1), Node(4)


if __name__ == "__main__":
    from time import perf_counter

    for _ in range(1):
        edges, g, V, E, source, sink = fill_graph_push_relabel(400, 200, 500)
        # edges, g, V, E, source, sink = fill_test_graph()
        # edges = [(1, 2, 1), (1, 4, 1), (2, 3, 21), (3, 4, 10), (4, 5, 68)]
        # source = 1
        # sink = 5
        # V = 6
        # E = 9
        # g = defaultdict(dict)
        print("|V| = %.2d, |E| = %.2d, s = %.d, t = %.d" % (V, E, source.id, sink.id))
        # g.dot(source, sink).render("graph", format="pdf", cleanup=True)
        results = []
        ss = (
            FifoPushRelabelSolver(g),
            RelabelToFrontSolver(g),
            EdmondsKarpSolver(g),
            DinicsSolver(g),
            CapacityScalingSolver(g),
        )
        for solver in ss:
            t = perf_counter()
            maxflow = solver.solve(source, sink)
            t = perf_counter() - t
            print(f"{solver.__class__.__name__}: {t:.5f} seconds, maxflow: {maxflow}")
            results.append(maxflow)
        if not (results[0] == results[1] == results[2] == results[3] == results[4]):
            ss[0].graph.dot(source, sink).render("graph0", format="pdf", cleanup=True)
            ss[1].graph.dot(source, sink).render("graph1", format="pdf", cleanup=True)
            pprint(g)
            print(RelabelToFrontSolver(g).solve(source, sink))
            pprint(edges)
            assert False
