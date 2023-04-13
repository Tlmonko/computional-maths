import numpy as np

from task2.subtask1.linear_system import solve_linear_system_with_lup


def inverse(matrix):
    n = matrix.shape[0]
    E = np.eye(n)
    X = np.zeros((n, n))
    for i in range(n):
        X[:, i] = solve_linear_system_with_lup(matrix, E[:, i])
    return X


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    A = np.random.randint(-50, 50, size=(n, n)).astype(float)

    X = inverse(A.copy())

    print('A:', A, sep='\n', end='\n\n')
    print('X:', X, sep='\n', end='\n\n')

    print('A * X == E:', np.allclose(A.dot(X), np.eye(n)))
    print('X * A == E:', np.allclose(X.dot(A), np.eye(n)))
