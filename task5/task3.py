from typing import Callable
from task1 import my_function, my_function_derivative
from common.utils import get_cheb_points
from common.draw_plot import draw_plot
from hermit_polynomial import get_hermite_polynomial
from terminaltables import AsciiTable
import numpy as np


def draw_hermite_polynomial_points(function: Callable, function_derivative: Callable, interpolation_points_count=5):
    interval = (-2, 2)

    x = np.linspace(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x], [function_derivative(i) for i in x])

    x_cheb = get_cheb_points(*interval, interpolation_points_count)
    cheb_interpolation_points = (x_cheb, [function(x) for x in x_cheb], [function_derivative(i) for i in x_cheb])

    hermite_polynomial = get_hermite_polynomial(*interpolation_points)
    hermite_polynomial_cheb = get_hermite_polynomial(*cheb_interpolation_points)

    draw_plot(interval, [interpolation_points[:-1], cheb_interpolation_points[:-1]],
              [function, lambda t: np.polyval(hermite_polynomial_cheb, t),
               lambda t: np.polyval(hermite_polynomial, t)],
              ['Primary function', 'Hermit chebyshev', 'Hermite'])


def find_hermit_polynomial_accuracy(function: Callable, function_derivative: Callable,
                                    get_points: Callable = np.linspace,
                                    interpolation_points_count=5):
    interval = (-2, 2)

    x = get_points(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x], [function_derivative(i) for i in x])

    polynomial = get_hermite_polynomial(*interpolation_points)
    polynomial_der = np.polyder(polynomial)

    hermite = lambda x: np.polyval(polynomial, x)
    hermite_der = lambda x: np.polyval(polynomial_der, x)

    accuracy_points_count = 200
    dx = (interval[1] - interval[0]) / accuracy_points_count
    return max(abs(hermite(x) - function(x)) * dx for x in np.linspace(*interval, num=accuracy_points_count)), max(
        abs(hermite_der(x) - function_derivative(x)) * dx for x in np.linspace(*interval, num=accuracy_points_count))


def draw_and_compare(function, function_derivative):
    draw_hermite_polynomial_points(function, function_derivative)

    table = [['n', 'P', 'Ch', 'P \'', 'Ch \'']]
    for n in range(3, 6):
        default_accuracy, default_der_accuracy = find_hermit_polynomial_accuracy(function, function_derivative)
        cheb_accuracy, cheb_der_accuracy = find_hermit_polynomial_accuracy(function, function_derivative,
                                                                           get_cheb_points)
        table.append([n, default_accuracy, cheb_accuracy, default_der_accuracy, cheb_der_accuracy])

    print(AsciiTable(table).table)


if __name__ == '__main__':
    draw_and_compare(my_function, my_function_derivative)
