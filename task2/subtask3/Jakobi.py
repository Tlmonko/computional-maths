import random

import numpy as np
import math

from Seidel import seidel


def jakobi(A, B):
    n = A.shape[0]
    res = np.zeros((n, 1))

    count = 0
    while count < 50:
        prv = np.copy(res)
        for k in range(n):
            s = sum(A[k][j] * res[j] for j in range(n) if j != k)
            res[k] = B[k] / A[k][k] - s / A[k][k]

        if np.linalg.norm(res - prv) < 1e-6:
            break
        count += 1

    return res


def get_matrix_with_diagonal_predominance(n):
    matrix = np.random.randint(-50, 50, size=(n, n)).astype(float)
    for i in range(n):
        matrix[i][i] = random.randint(50 * n ** 2, 100 * n ** 2)
    return matrix


def get_positive_definite_matrix(n):
    return np.random.randint(0, 50, size=(n, n)).astype(float)


def get_priority_assessment(matrix):
    epsilon = 10 ** (-4)
    n = matrix.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i][i] = matrix[i][i]

    E = np.eye(n)
    B = E - np.dot(np.linalg.inv(D), matrix)
    C = np.dot(np.linalg.inv(D), b)

    bnorm = np.linalg.norm(B)
    cnorm = np.linalg.norm(C)
    if bnorm < 1:
        math.ceil(math.log((epsilon / cnorm * (1 - bnorm)), bnorm))
    if bnorm > 1:
        return math.ceil(math.log((epsilon / cnorm * (1 - 1e-1)), 1e-1))


if __name__ == '__main__':
    n = np.random.randint(2, 10)
    diagonal_predominance_matrix = get_matrix_with_diagonal_predominance(n)
    positive_matrix = get_positive_definite_matrix(n)
    b = np.random.rand(n, 1)

    print(jakobi(diagonal_predominance_matrix, b))
    print(jakobi(positive_matrix, b))

    print(seidel(diagonal_predominance_matrix, b))
    print(seidel(positive_matrix, b))

    print(get_priority_assessment(diagonal_predominance_matrix))



