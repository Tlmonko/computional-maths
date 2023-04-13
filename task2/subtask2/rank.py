import numpy as np
from task2.subtask1.lup_decomposition import lup_decomposition


def get_rank(matrix: np.ndarray) -> int:
    L, u, _, _ = lup_decomposition(matrix)
    print(L, u)
    return sum(np.any(u, axis=1))


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    # A = np.random.randint(-50, 50, size=(n, n)).astype(float)
    A = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    print('A:', A, sep='\n', end='\n\n')

    print('rank(A):', get_rank(A))
    print('rank(A) == np.linalg.matrix_rank(A):', np.isclose(get_rank(A), np.linalg.matrix_rank(A)))
