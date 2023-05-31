import gc

from capacity_scaling import CapacityScaler
from dinitz import LayeredGraph
from edmonds_karp import FlowNetwork
from push_relabel import *


def graph_dinitz():
    graph = LayeredGraph()
    return (graph,) + fill(graph)


def graph_edmonds():
    graph = FlowNetwork()
    return (graph,) + fill(graph)


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
    g = Digraph()
    e = gen_net(nn, b, max_cap)
    for edge in e:
        g.insert_edge(*edge)
    return e, g, nn, len(e), Node(1), Node(nn + 1)


if __name__ == "__main__":
    from time import perf_counter

    edges, _, V, E, source, sink = fill_graph_push_relabel(400, 200, 500)
    # edges = [(1, 2, 1), (1, 4, 1), (2, 3, 21), (3, 4, 10), (4, 5, 68)]
    # source = 1
    # sink = 5
    # V = 6
    # E = 9
    # g = defaultdict(dict)
    print("|V| = %.2d, |E| = %.2d, s = %.d, t = %.d" % (V, E, source.id, sink.id))
    g = FifoPushRelabel()
    g.insert_edges_from_iterable(edges)
    t1 = perf_counter()
    maxflow1 = g.run(source, sink)
    t1 = perf_counter() - t1
    print("fifo: %.5f seconds" % t1)

    g = RelabelToFront()
    g.insert_edges_from_iterable(edges)
    t2 = perf_counter()
    maxflow2 = g.run(source, sink)
    t2 = perf_counter() - t2
    print("relabel-to-front: %.5f seconds" % t2)

    g = FlowNetwork()
    g.insert_edges_from_iterable(edges)
    t3 = perf_counter()
    maxflow3 = g.edmonds_karp(source, sink)
    t3 = perf_counter() - t3
    print("edmonds-karp: %.5f seconds" % t3)

    g = LayeredGraph()
    g.insert_edges_from_iterable(edges)
    t4 = perf_counter()
    maxflow4 = g.dinitz_algorithm(source, sink)
    t4 = perf_counter() - t4
    print("dinic: %.5f seconds" % t4)

    g = CapacityScaler()
    g.insert_edges_from_iterable(edges)
    t5 = perf_counter()
    maxflow5 = g.find_max_flow(source, sink)
    t5 = perf_counter() - t5
    print("capacity scaling: %.5f seconds" % t5)
    print(maxflow1, maxflow2, maxflow3, maxflow4, maxflow5)
    assert all(m == maxflow1 for m in (maxflow2, maxflow3, maxflow4, maxflow5))
    gc.collect()
