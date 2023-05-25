import numpy as np
from typing import List


def __get_matrix(x: List[float], second_der=False):
    n = len(x)
    m = n * 3 - 1 if second_der else n * 2 - 1
    matrix = [[x[i] ** k for k in range(m + 1)] for i in range(n)]
    matrix.extend([[0.0] + [(k + 1) * x[i] ** k for k in range(m)] for i in range(n)])
    if second_der:
        matrix.extend([[0.0, 0.0] + [k * (k - 1) * x[i] ** (k - 2) for k in range(2, m + 1)] for i in range(n)])
    return matrix


def __get_free_coeffs(y: List[float], y_derivative: List[int], y_second_derivative=None):
    y = y.copy()
    y.extend(y_derivative)
    if y_second_derivative is not None:
        y.extend(y_second_derivative)

    return y


def get_hermite_polynomial(x: List[float], y: List[float], y_derivative: List[int], y_second_derivative=None):
    matrix = __get_matrix(x, second_der=y_second_derivative is not None)
    b = __get_free_coeffs(y, y_derivative, y_second_derivative)


    coefficients = np.linalg.solve(matrix, b)

    return coefficients[::-1]
