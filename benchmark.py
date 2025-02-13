from time import perf_counter

from rich import print as rprint

from solvers import *
from utils import gen_random_network


def run():
    flow_network, source, sink = gen_random_network(200, 100, 500)
    rprint(
        f"Flow network with {flow_network.n_nodes} nodes and {flow_network.n_edges()} edges."
    )
    for solver_class in (
        FifoPushRelabelSolver,
        RelabelToFrontSolver,
        EdmondsKarpSolver,
        CapacityScalingSolver,
        DinicsSolver,
        LinProgSolver,
    ):
        start = perf_counter()
        solver = solver_class(flow_network)
        maxflow = solver.solve(source, sink)
        end = perf_counter()
        rprint(f"{solver_class.__name__}: {end - start} seconds")
        rprint(f"Max flow: {maxflow}")
        # solver.graph.check_flow_conservation(source, sink)


if __name__ == "__main__":
    run()
