import numpy as np

from task2.subtask1.lup_decomposition import lup_decomposition


def solve_upper_triangular_system(matrix, b):
    n = matrix.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(matrix[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / matrix[i, i]
    return x


def solve_lower_triangular_system(matrix, b):
    n = matrix.shape[0]
    x = np.zeros(n)
    for i in range(n):
        s = sum(matrix[i, j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / matrix[i, i]
    return x


def solve_linear_system_with_lup(matrix, b):
    L, U, P, _ = lup_decomposition(matrix)
    y = solve_lower_triangular_system(L, P.dot(b))
    return solve_upper_triangular_system(U, y)


def solve_linear_system_with_knowing_lup(L, U, P, b):
    y = solve_lower_triangular_system(L, P.dot(b))
    return solve_upper_triangular_system(U, y)


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    A = np.random.randint(-50, 50, size=(n, n)).astype(float)
    b = np.random.randint(-50, 50, size=n).astype(float)

    # A = np.array([[1, 1, 1], [1, -1, 0], [0, 1, -1]]).astype(float)
    # b = np.array([3, 0, 0]).astype(float)

    x = solve_linear_system_with_lup(A, b)

    print('A:', A, sep='\n', end='\n\n')
    print('b:', b, sep='\n', end='\n\n')
    print('x:', x, sep='\n', end='\n\n')

    print('A * x == b:', np.allclose(A.dot(x), b))

