from collections import defaultdict
from heapq import heapify, heappop, heappush
from typing import Iterable, Optional, TypeVar

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


Item = TypeVar("Item")


class MinHeapSet(list[tuple[float, Item]]):
    def __init__(self, items_iterable: Optional[Iterable[tuple[float, Item]]] = None):
        super().__init__()
        if items_iterable is not None:
            items = tuple(items_iterable)
            self._set = {item for _, item in items}
            if items is not None:
                self.extend(items)
                heapify(self)
        else:
            self._set = set()

    def add(self, item: Item, priority: float):
        self._set.add(item)
        heappush(self, (priority, item))

    def extract(self) -> Item:
        priority, item = heappop(self)
        self._set.remove(item)
        return item

    def __contains__(self, item):
        return item in self._set


class UnionFind:
    """A simple implementation of a disjoint-set data structure.
    The amortized running time is O(m ⍺(n)) for m disjoint-set operations on n elements, where
    ⍺(n) is the inverse Ackermann function .⍺(n) grows extremely slowly and can be assumed to be ⩽ 5 for
    all practical purposes.
    Operations:
        MAKE-SET(x) – creates a new set with one element {x}.
        UNION(x, y) – merge into one set the set that contains element x and the set that contains element y (x and y are in different sets). The original sets will be destroyed.
        FIND-SET(x) – returns the representative or a pointer to the representative of the set that contains element x.
    Applications of UnionFind include:
        1. Kruskal’s algorithm for MST.
        2. They are useful in applications like “Computing the shorelines of a terrain,”
            “Classifying a set of atoms into molecules or fragments,” “Connected component labeling in image analysis,” and others.[1]
        3. Labeling connected components.
        4. Random maze generation and exploration.
        5. Alias analysis in compiler theory.
        6. Maintaining the connected components of an undirected-graph, when the edges are being added dynamically.
        7. Strategies for games: Hex and Go.
        8. Tarjan's offline Least common ancestor algorithm.
        9. Cycle detection in undirected graph.
        10. Equivalence of finite state automata

    Reference:
        1.  Cormen, Leiserson, Rivest, Stein,. "Chapter 21: Data structures for Disjoint Sets".
            Introduction to Algorithms (Third ed.). MIT Press. pp. 571–572. ISBN 978-0-262-03384-8.
        2.  https://www.topcoder.com/community/competitive-programming/tutorials/disjoint-set-data-structures/
        3.  https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        4.  https://www.cs.upc.edu/~mjserna/docencia/grauA/T19/Union-Find.pdf

    """

    def __init__(self, items=None):
        # MAKE-SET()
        # we don't need to store the elements, instead we can hash them, and since each element is unique, then
        # hashes won't collide. I will use union by size instead of union by rank
        # using union by rank needs more careful handling in the union of multiple items

        if items is None:
            items = ()
        self.parents = {}
        self.weights = {}

        for item in items:
            self.parents[item] = item
            self.weights[item] = 0

    def __getitem__(self, item):
        # FIND-SET()
        if item not in self.parents:
            self.parents[item] = item  # MAKE-SET()
            self.weights[item] = 0
            return item
        else:
            # store nodes in the path leading to the root(representative) for later updating
            # this is the path-compression step
            path = [item]
            root = self.parents[item]
            while root != path[-1]:
                path.append(root)
                root = self.parents[root]
            for node in path:
                self.parents[node] = root
            return root

    def union(self, *items):
        # UNION()
        roots = [self[item] for item in items]
        # find the root with the largest weight and the other roots to it
        heaviest = max(roots, key=lambda item: self.weights[item])
        for root in roots:
            if root != heaviest:
                self.parents[root] = heaviest
                self.weights[heaviest] += self.weights[root]

    def __iter__(self):
        return iter(self.parents)

    def _groups(self):
        one_to_many = defaultdict(set)
        for v, k in self.parents.items():
            one_to_many[k].add(v)
        return dict(one_to_many)

    def to_sets(self):
        yield from self._groups().values()

    def __str__(self):
        return str(list(self.to_sets()))
