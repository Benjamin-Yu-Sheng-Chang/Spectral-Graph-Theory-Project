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

def generate_symmetric_adjacency_matrices_3_eigenvalues(n, self_loops=True, tolerance=1e-9):
    """
    Generate all non-similar symmetric adjacency matrices of size n with self-loops
    that have exactly three distinct eigenvalues.
    
    Parameters:
    -----------
    n : int
        Size of the adjacency matrix (n x n)
    tolerance : float
        Tolerance for determining if eigenvalues are distinct (default: 1e-9)
    
    Returns:
    --------
    list of numpy.ndarray
        List of non-similar n x n symmetric adjacency matrices with exactly 3 distinct eigenvalues
    """
    if n < 1:
        raise ValueError("Matrix size n must be at least 1")
    
    # For symmetric matrix, we only need upper triangular entries (including diagonal)
    num_entries = n * (n + 1) // 2
    
    # First, collect all matrices with 3 eigenvalues
    candidates = []
    
    # Iterate through all 2^num_entries possible configurations
    for config in product([0, 1], repeat=num_entries):
        # Build symmetric matrix from configuration
        matrix = np.zeros((n, n), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = config[idx]
                matrix[j, i] = config[idx]  # Ensure symmetry
                idx += 1

        if not self_loops:
            for i in range(n):
                matrix[i, i] = 0
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)
        
        # Round eigenvalues to handle numerical precision
        rounded_eigenvalues = np.round(eigenvalues / tolerance) * tolerance
        
        # Count distinct eigenvalues
        unique_eigenvalues = np.unique(rounded_eigenvalues)
        
        if len(unique_eigenvalues) == 3:
            candidates.append(matrix)
    
    # Filter out similar matrices
    non_similar_matrices = []
    
    for candidate in candidates:
        is_similar_to_existing = False
        
        # Check if candidate is similar to any already selected matrix
        for selected in non_similar_matrices:
            if are_similar(candidate, selected, tolerance):
                is_similar_to_existing = True
                break
        
        if not is_similar_to_existing:
            non_similar_matrices.append(candidate)
    
    return non_similar_matrices


def draw_with_big_loops(A, loop_size=0.3):
    G = nx.from_numpy_array(A, create_using=nx.Graph)

    pos = nx.spring_layout(G, seed=1)

    # Draw nodes and normal edges
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=[
        e for e in G.edges() if e[0] != e[1]
    ])

    # Draw self-loops manually as circles
    ax = plt.gca()
    for n in G.nodes():
        if G.has_edge(n, n):
            x, y = pos[n]
            circle = plt.Circle(
                (x, y),
                radius=loop_size,
                fill=False,
                linewidth=2
            )
            ax.add_patch(circle)

    plt.axis("off")
    plt.show()