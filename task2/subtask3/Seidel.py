import numpy as np


def seidel(matrix, b):
    epsilon = 10e-6
    n = matrix.shape[0]
    x = np.zeros((n, 1))
    while True:
        new_x = np.copy(x)
        s1 = 0
        s2 = 0
        for i in range(n):
            for j in range(i):
                s1 += matrix[i][j] * new_x[j]
            for j in range(i + 1, n):
                s2 += matrix[i][j] * x[j]
            new_x[i] = (b[i] - s1 - s2) / matrix[i][i]
        if np.linalg.norm(new_x - x) <= epsilon:
            break
        x = new_x
    return x
