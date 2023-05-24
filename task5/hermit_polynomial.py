import numpy as np
from typing import List


def __get_matrix(x: List[float]):
    n = len(x)
    m = n * 2 - 1
    matrix = [[x[i] ** k for k in range(m + 1)] for i in range(n)]
    matrix.extend([[0.0] + [(k + 1) * x[i] ** k for k in range(m)] for i in range(n)])
    return matrix


def __get_free_coeffs(y: List[float], y_derivative: List[int]):
    y = y.copy()
    y.extend(y_derivative)
    return y


def get_hermite_polynomial(x: List[float], y: List[float], y_derivative: List[int]):
    matrix = __get_matrix(x)
    b = __get_free_coeffs(y, y_derivative)

    coefficients = np.linalg.solve(matrix, b)

    return coefficients[::-1]
