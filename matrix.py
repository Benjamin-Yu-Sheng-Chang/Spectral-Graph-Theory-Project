import numpy as np
from itertools import product, permutations
import networkx as nx
import matplotlib.pyplot as plt

def are_similar(A, B, tolerance=1e-9):
    """
    Check if matrices A and B are similar (i.e., B = P^(-1) * A * P for some P).
    For adjacency matrices, we check similarity via permutation matrices.
    """
    n = A.shape[0]
    
    # Try all permutations
    for perm in permutations(range(n)):
        P = np.eye(n)[list(perm), :]  # Permutation matrix
        # Check if P^T * A * P = B (for permutation matrices, P^(-1) = P^T)
        transformed = P.T @ A @ P
        if np.allclose(transformed, B, atol=tolerance):
            return True
    return False


def full_bipartite_graph(n, m):
    zero_nn = np.zeros((n, n))
    zero_mm = np.zeros((m, m))
    ones_nm = np.ones((n, m))
    ones_mn = np.ones((m, n))
    
    return np.block([
        [zero_nn,  ones_nm],
        [ones_mn,  zero_mm]
    ])


def srg_quotient_matrix(k, _lambda, mu):
    return np.array([
        [0, k, 0], 
        [1, _lambda, k - _lambda - 1], 
        [0, mu, k - mu]])

def srg_quotient_matrix__one_self_loop(k, _lambda, mu):
    return np.array([[1, k, 0], [1, _lambda, k - _lambda - 1], [0, mu, k - mu]])


def permute_matrix_by_partition(matrix, partition):
    """
    Permute the rows and columns of a matrix so that vertices in the same
    partition cell are grouped together. Cell 0 comes first, then cell 1, etc.

    Parameters:
    -----------
    matrix : array-like
        Square matrix (e.g. adjacency matrix) of shape (n, n).
    partition : array-like
        Array of length n where partition[i] is the cell index of vertex i.

    Returns:
    --------
    permuted_matrix : np.ndarray
        Matrix with rows and columns reordered by partition.
    perm : np.ndarray
        The permutation index array such that permuted_matrix = matrix[perm][:, perm].
    """
    matrix = np.asarray(matrix)
    partition = np.asarray(partition)
    n = matrix.shape[0]
    if partition.shape[0] != n:
        raise ValueError("partition length must equal matrix size n")
    # Indices that sort by partition: all cell-0 vertices first, then cell-1, etc.
    perm = np.argsort(partition)
    return matrix[np.ix_(perm, perm)].copy(), perm


def get_least_cell_quotient(adj_matrix):
    """
    Compute the quotient matrix for the coarsest equitable partition of a graph.
    
    An equitable partition is a partition where the number of edges from a node
    to cells depends only on which cell the node is in, not the specific node.
    
    Parameters:
    -----------
    adj_matrix : array-like
        The adjacency matrix of the graph
    
    Returns:
    --------
    quotient_matrix : np.ndarray
        The quotient matrix (each entry is the number of edges between cells)
    partition : np.ndarray
        The partition of vertices (cell assignment for each vertex)
    """
    adj = np.array(adj_matrix)
    n = adj.shape[0]
    
    # 1. Start with all nodes in the same partition (color 0)
    partition = np.zeros(n, dtype=int)
    
    while True:
        # 2. For each node, create a signature based on its neighbors' partitions
        # Signature = (current_partition, sorted_list_of_neighbor_partitions)
        signatures = []
        for i in range(n):
            neighbor_parts = sorted(partition[j] for j in range(n) if adj[i, j] > 0)
            signatures.append((partition[i], tuple(neighbor_parts)))
        
        # 3. Map unique signatures to new partition labels (colors)
        unique_sigs = sorted(list(set(signatures)))
        sig_map = {sig: idx for idx, sig in enumerate(unique_sigs)}
        new_partition = np.array([sig_map[sig] for sig in signatures])
        
        # 4. If the partition hasn't changed, we've found the coarsest equitable partition
        if np.array_equal(new_partition, partition):
            break
        partition = new_partition

    # 5. Build the quotient matrix
    num_cells = len(unique_sigs)
    quotient_matrix = np.zeros((num_cells, num_cells))
    
    cells = [np.where(partition == i)[0] for i in range(num_cells)]
    
    for i in range(num_cells):
        for j in range(num_cells):
            # Pick any representative node 'u' from cell i
            u = cells[i][0]
            # Count edges from 'u' to all nodes in cell j
            edge_count = sum(adj[u, v] for v in cells[j])
            quotient_matrix[i, j] = edge_count
            
    return quotient_matrix, partition


def np_unique(A):
    eigenvalues = np.linalg.eigvalsh(A)
    rounded_eigenvalues = np.round(eigenvalues / tolerance) * tolerance
    unique_eigenvalues = np.unique(rounded_eigenvalues)
    return unique_eigenvalues