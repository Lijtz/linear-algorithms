# gaussian_elimination.py
"""
Gaussian elimination (forward elimination) returning row echelon form.
Input: augmented matrix (m x (n+1)) as numpy array or nested list
Output: row echelon form as numpy array
"""

import numpy as np

def to_numpy(A):
    A = np.array(A, dtype=float)
    return A

def row_echelon(A):
    """
    Parameters:
        A : array-like, shape (m, n+1) augmented matrix
    Returns:
        U : numpy.ndarray row-echelon form of A (float)
    """
    A = to_numpy(A)
    m, n = A.shape
    row = 0
    for col in range(n - 1): 
        pivot_row = np.argmax(np.abs(A[row:m, col])) + row
        if abs(A[pivot_row, col]) < 1e-12:  
            continue
        if pivot_row != row:
            A[[row, pivot_row], :] = A[[pivot_row, row], :]
        for r in range(row + 1, m):
            if A[r, col] == 0:
                continue
            factor = A[r, col] / A[row, col]
            A[r, col:] = A[r, col:] - factor * A[row, col:]
        row += 1
        if row == m:
            break
    A[np.abs(A) < 1e-12] = 0.0
    return A

if __name__ == "__main__":
    A = [
        [2, 1, -1,  8],
        [-3, -1, 2, -11],
        [-2, 1, 2, -3]
    ] 
    U = row_echelon(A)
    print("Row echelon form:\n", U)
