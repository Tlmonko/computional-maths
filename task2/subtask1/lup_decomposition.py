import numpy as np
from typing import Tuple


def lup_decomposition(matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    matrix = matrix.copy().astype(float)
    n = matrix.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)
    count_of_swaps = 0
    is_remain_non_zero = True

    for i in range(n):
        if not is_remain_non_zero:
            break
        max_element = np.max(np.abs(matrix[i:, i]))
        max_element_indexes = np.where(np.abs(matrix[i:, i]) == max_element)
        max_element_index = max_element_indexes[0][0] + i

        matrix[[i, max_element_index]] = matrix[[max_element_index, i]]
        P[[i, max_element_index]] = P[[max_element_index, i]]
        if i != max_element_index:
            count_of_swaps += 1

        for j in range(i + 1, n):
            if matrix[i, i] == 0:
                last_non_zero_column = -1
                for k in range(n - 1, i, -1):
                    if np.any(matrix[i:, k]):
                        last_non_zero_column = k
                if last_non_zero_column == -1:
                    is_remain_non_zero = False
                    break
                matrix[:, [i, last_non_zero_column]] = matrix[:, [last_non_zero_column, i]]
            matrix[j, i] /= matrix[i, i]
            for k in range(i + 1, n):
                matrix[j, k] -= matrix[j, i] * matrix[i, k]

    for i in range(n):
        for j in range(n):
            if j < i:
                L[i, j] = matrix[i, j]
            else:
                U[i, j] = matrix[i, j]

    return L, U, P, (-1) ** count_of_swaps


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    A = np.random.randint(-50, 50, size=(n, n)).astype(float)
    # A = np.array([[1, 2, 0], [3, 5, 4], [5, 6, 3]]).astype(float)
    # A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]).astype(float)

    L, U, P, P_det = lup_decomposition(A.copy())

    print('A:', A, sep='\n', end='\n\n')
    print('L:', L, sep='\n', end='\n\n')
    print('U:', U, sep='\n', end='\n\n')
    print('P:', P, sep='\n', end='\n\n')

    print()

    print('L * U == P * A:', np.allclose(L.dot(U), P.dot(A)))
