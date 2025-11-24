# gram_schmidt.py
"""
Modified Gram-Schmidt orthonormalization.
Input: A (m x n) matrix with vectors as columns (list or numpy array)
Output: Q (m x n) orthonormal columns, R (n x n) upper triangular
"""

import numpy as np

def modified_gram_schmidt(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 1e-12:
            raise np.linalg.LinAlgError(f"Column {i} is linearly dependent or zero")
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R

if __name__ == "__main__":
    # Example
    A = np.array([[1., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 1.]])
    Q, R = modified_gram_schmidt(A)
    print("Q:\n", Q)
    print("R:\n", R)
    # Verify A == Q @ R
    print("Q @ R:\n", Q @ R)
_
