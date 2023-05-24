from typing import List
import numpy as np
from common.utils import __find__range_index


def __get_spline_31_coefficients(x: List[float], y: List[float], y_der: List[float], k):
    dx = x[k] - x[k - 1]
    a0 = y[k - 1]
    a1 = y_der[k - 1]
    a2 = (3 * y[k] - 3 * y[k - 1] - 2 * dx * y_der[k - 1] - dx * y_der[k]) / (dx ** 2)
    a3 = (2 * y[k - 1] - 2 * y[k] + dx * y_der[k - 1] + dx * y_der[k]) / (dx ** 3)
    return [a3, a2 - 3 * x[k - 1] * a3, a1 - 2 * a2 * x[k - 1] + 3 * a3 * (x[k - 1] ** 2),
            a0 - a1 * x[k - 1] + a2 * (x[k - 1] ** 2) - a3 * (x[k - 1] ** 3)]


def get_spline_31(x, x_coords: List[float], y_coords: List[float], y_der: List[float]):
    i = __find__range_index(x, x_coords)
    if i == 0:
        i = 1

    return np.polyval(__get_spline_31_coefficients(x_coords, y_coords, y_der, i), x)
