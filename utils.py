import numpy as np

from core import FlowNetwork, Node


def gen_random_network_helper(
    num_nodes: int, max_outgoing: int, max_cap: int
) -> list[tuple[Node, Node, int]]:
    num_nodes += 1
    edges = []
    for u in range(1, num_nodes):
        out = set(
            np.random.randint(
                u + 1,
                num_nodes + 1,
                np.random.randint(max_outgoing >> 1, max_outgoing),
            )
        )
        for v in out:
            edges.append((Node(u), Node(v), np.random.randint(0, max_cap)))
    return edges


def gen_random_network(
    num_nodes: int, max_outgoing: int, max_cap: int
) -> tuple[FlowNetwork, Node, Node]:
    edges = gen_random_network_helper(num_nodes, max_outgoing, max_cap)
    flow_network = FlowNetwork()
    for edge in edges:
        flow_network.insert_edge(*edge)
    return flow_network, Node(1), Node(num_nodes + 1)
