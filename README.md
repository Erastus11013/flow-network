# Network Flow Algorithms

We are given a directed graph *G*, a start node *s*, and a sink node *t*. Each edge *(u, v)* in G has an associated non-negative capacity *c(u, v)*, where for all non-edges it is implicitly assumed that the capacity is 0. Our goal is to push as much flow as possible from s to t in the graph. The rules are that no edge can have flow exceeding its capacity, and for any vertex except for s and t, the flow in to the vertex must equal the flow out from the vertex.

# Algorithms:
- Edmonds-Karp implementation of the Ford-Fulkerson method.
- Dinitz Algorithm

# References
[Data Structures and Network Algorithms](https://dl.acm.org/citation.cfm?id=3485)
