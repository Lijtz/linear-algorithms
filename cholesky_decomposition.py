# cholesky_decomposition.py
"""
Cholesky decomposition (lower-triangular L such that A = L @ L.T).
Input: symmetric positive-definite matrix A (numpy array or nested list).
Output: L (numpy array)
"""

import numpy as np

def cholesky(A):
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    # Check symmetry
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Matrix is not symmetric")
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - np.dot(L[i, :j], L[j, :j])
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L

if __name__ == "__main__":
    # Example
    A = [[25, 15, -5],
         [15, 18,  0],
         [-5,  0, 11]]
    L = cholesky(A)
    print("L:\n", L)
    # Verify
    print("L @ L.T:\n", L @ L.T)
