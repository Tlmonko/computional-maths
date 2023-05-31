import numpy as np
from common.draw_plot import draw_plot
from task1 import my_function
import scipy
from typing import List


def __get_lezhandr_poynomial(n, x):
    if n == 0:
        return 1
    if n == 1:
        return x
    return ((2 * n - 1) * x * __get_lezhandr_poynomial(n - 1, x) - (n - 1) * __get_lezhandr_poynomial(n - 2, x)) / n


def get_approximation_polynomial(n, f):
    matrix = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for k in range(n + 1):
        for j in range(n + 1):
            matrix[k][j] = \
                scipy.integrate.quad(lambda x: __get_lezhandr_poynomial(k, x) * __get_lezhandr_poynomial(j, x), -1, 1)[
                    0]

    for i in range(n + 1):
        b[i] = scipy.integrate.quad(lambda x: f(x) * __get_lezhandr_poynomial(i, x), -1, 1)[0]
    return np.linalg.solve(matrix, b)


def __get_polynomial(x: float, coeffs: List[float]):
    return sum(__get_lezhandr_poynomial(i, x) * coeffs[i] for i in range(len(coeffs)))


if __name__ == '__main__':
    n = 6
    interval = (-2, 2)

    x_points = np.linspace(*interval, n)
    points = (x_points, [my_function(x) for x in x_points])

    lezhandr_coeffs = get_approximation_polynomial(n, my_function)

    draw_plot(interval, [points], [my_function, lambda x: __get_polynomial(x, lezhandr_coeffs)],
              ['Primary function', 'approximation'])
