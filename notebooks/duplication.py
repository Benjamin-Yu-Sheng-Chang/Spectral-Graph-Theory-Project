from pathlib import Path
import sys

from notebooks.test import A1

sys.path.append(str(Path().cwd().parent))
import networkx as nx
import numpy as np
from matrix import are_similar, get_least_cell_quotient, np_unique, permute_matrix_by_partition
from itertools import combinations
from numpy import trace
from networkx import is_bipartite

tolerance = 1e-10

np.linalg.eigvals(
    np.array([
        [0, 1, 1],
        [1, 3, 1],
        [4, 4, 1]
    ])
)

np_unique(
    np.array([
        [0,2],[2,3]
    ])
)

np.linalg.eigvals(
    np.array(
        [[1., 2.],
        [4., 3.]]
    )
)

np.linalg.eigvals(
    np.array(
        [[2., 2.],
        [6., 3.]]
    )
)

A = np.zeros((3, 3))
B = np.eye(3)
C = np.array(
    [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
)
D = np.ones((3, 3))

E = np.block([
    [A, B, B, C],
    [B, A, B, C],
    [B, B, A, C],
    [C, C, C, D]
])

E2 = np.block([
    [A, B, B, B, C],
    [B, A, B, B, C],
    [B, B, A, B, C],
    [B, B, B, A, C],
    [C, C, C, C, D]
])

np_unique(E2)
nx.draw(nx.from_numpy_array(E2), with_labels=True)

np.linalg.eigvals(
    np.array(
        [[2., 1.],
        [2., 3.]]
    )
)

# not constructible
np.linalg.eigvals(
    np.array(
        [[1., 1.],
        [0., 3.]]
    )
)

np.linalg.eigvals(
    np.array(
        [[3., 1.],
        [4., 3.]]
    )
)

def cycle_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    return A

def duplicate(A: np.ndarray, n: int):
    return np.vstack([A for _ in range(n)])

A = cycle_matrix(6)
B = np.vstack([np.eye(3), np.eye(3)])
C = np.ones((3, 3))

E = np.block([
    [A, B],
    [B.T, C]
])


np_unique(E)
nx.draw(nx.from_numpy_array(E), with_labels=True)

q_mat, part = get_least_cell_quotient(E)
permuted, perm = permute_matrix_by_partition(E, part)

def geneerate_line_graph(A):
    G = nx.from_numpy_array(A)
    L_G = nx.line_graph(G)
    return nx.to_numpy_array(L_G)

nx.draw(nx.from_numpy_array(geneerate_line_graph(permuted)), with_labels=True)

E2 = np.block([
    [cycle_matrix(9), np.vstack([np.eye(3) for _ in range(3)])],
    [np.hstack([np.eye(3) for _ in range(3)]), np.ones((3, 3))]
])

np_unique(E2)
q_mat
nx.draw(nx.from_numpy_array(permuted), with_labels=True)

np.linalg.eigvals(
    np.array(
        [
        [1, 2],
        [4, 3]
        ]
    )
)


np.linalg.eigvals(
    np.array(
        [
        [2, 1],
        [2, 3]
        ]
    )
)

