import numpy as np
from typing import List
from spline_31 import get_spline_31


def __get_c(y_coords: List[float], h: float, i: int):
    return 1.5 / h * (y_coords[i + 1] - y_coords[i - 1])


def __get_m_coeff(i, j):
    if i == j:
        return 2
    elif abs(i - j) == 1:
        return 1
    return 0


def get_spline_32(x_coords: List[float], y_coords: List[float], der_0: float, der_1: float, h: float):
    n = len(x_coords)
    matrix = [[1.0] + [0.0] * (n - 1)] + \
             [[__get_m_coeff(i, x) for i in range(n)] for x in range(1, n - 1)] + \
             [[0.0] * (n - 1) + [1.0]]

    b = [der_0] + [__get_c(y_coords, h, i) for i in range(1, n - 1)] + [der_1]

    derivative_points = np.linalg.solve(matrix, b)

    return lambda t: get_spline_31(t, x_coords, y_coords, derivative_points)
