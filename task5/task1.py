import numpy as np
from lagrange_polynomial import get_lagrange_polynomial
from hermit_polynomial import get_hermite_polynomial
from common.draw_plot import draw_plot
from common.utils import get_cheb_points
from typing import Callable
from terminaltables import AsciiTable


def my_function(x):
    return np.tan(x / 2 + 0.2) - x ** 2 + 2


def my_function_derivative(x):
    return 1 / 2 * (np.cos(x / 2 + 0.2) ** 2) - 2 * x


def draw_lagrange_polynomial_points(function: Callable, function_derivative: Callable, interpolation_points_count=5):
    interval = (-2, 2)

    x = np.linspace(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x])

    x_cheb = get_cheb_points(*interval, interpolation_points_count)
    cheb_interpolation_points = (x_cheb, [function(x) for x in x_cheb])

    lagrange_polynomial = get_lagrange_polynomial(*interpolation_points)
    lagrange_polynomial_cheb = get_lagrange_polynomial(*cheb_interpolation_points)

    hermite_polynomial = get_hermite_polynomial(x, [function(i) for i in x], [function_derivative(i) for i in x])

    draw_plot(interval, [interpolation_points, cheb_interpolation_points, interpolation_points, interpolation_points],
              [function, lambda x: np.polyval(lagrange_polynomial_cheb, x),
               lambda x: np.polyval(lagrange_polynomial, x),
               lambda x: np.polyval(hermite_polynomial, x)],
              ['Primary function', 'Lagrange', 'Lagrange chebyshev', 'Hermite'])


def find_polynomial_accuracy(function: Callable, interpolation_function: Callable, get_points: Callable = np.linspace,
                             interpolation_points_count=5):
    interval = (-2, 2)

    x = get_points(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x])

    polynomial = interpolation_function(*interpolation_points)

    lagrange = lambda x: np.polyval(polynomial, x)

    accuracy_points_count = 200
    dx = (interval[1] - interval[0]) / accuracy_points_count
    return max(abs(lagrange(x) - function(x)) * dx for x in np.linspace(*interval, num=accuracy_points_count))


def draw_and_compare(function, function_derivative):
    draw_lagrange_polynomial_points(function, function_derivative)

    table = [['n', 'P', 'Ch']]
    table.extend([n, find_polynomial_accuracy(function, get_lagrange_polynomial),
                  find_polynomial_accuracy(function, get_lagrange_polynomial, get_cheb_points)] for n in range(3, 13))
    print(table)
    print(AsciiTable(table).table)


if __name__ == '__main__':
    draw_and_compare(my_function, my_function_derivative)
