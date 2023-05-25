from typing import Callable
from task1 import my_function, my_function_derivative
from common.utils import get_cheb_points, get_polynomial_accuracy
from common.draw_plot import draw_plot
from hermit_polynomial import get_hermite_polynomial
from lagrange_polynomial import get_lagrange_polynomial
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


def get_comparison_table(function, function_derivative, interpolating_function=get_hermite_polynomial):
    table = [['n', 'P', 'Ch', 'P \'', 'Ch \'']]
    for n in range(3, 6):
        default_accuracy, default_der_accuracy = get_polynomial_accuracy(function, function_derivative,
                                                                         interpolating_function, n)
        cheb_accuracy, cheb_der_accuracy = get_polynomial_accuracy(function, function_derivative,
                                                                   interpolating_function, n,
                                                                   get_cheb_points)
        table.append([n, default_accuracy, cheb_accuracy, default_der_accuracy, cheb_der_accuracy])

    return table


def get_comparison_table_same_degrees(function, function_derivative):
    hermit_accuracy = get_polynomial_accuracy(function, function_derivative, get_hermite_polynomial, 3)[0]
    lagrange_accuracy = get_polynomial_accuracy(function, function_derivative, get_lagrange_polynomial, 5)[0]
    return [['degree', 'Lagrange', 'Hermite'], [5, lagrange_accuracy, hermit_accuracy]]


if __name__ == '__main__':
    draw_hermite_polynomial_points(my_function, my_function_derivative)
    table = get_comparison_table(my_function, my_function_derivative)
    extend = get_comparison_table(my_function, my_function_derivative, get_lagrange_polynomial)

    print('Сравнение точности полиномов Лагранжа и Эрмита, а так же точности их производных')
    table[0] = ['n', 'Hermite P', 'Hermite Ch', 'Hermite P \'', 'Hermite Ch \'', 'Lagrange P', 'Lagrange Ch',
                'Lagrange P \'',
                'Lagrange Ch \'']
    for i in range(1, len(table)):
        table[i] += extend[i][1:]
    print(AsciiTable(table).table)

    print('Сравнение точности полиномов Лагранжа и Эрмита одинаковых степеней')
    table_same_degree = get_comparison_table_same_degrees(my_function, my_function_derivative)
    print(AsciiTable(table_same_degree).table)
