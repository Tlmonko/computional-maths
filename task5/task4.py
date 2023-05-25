from task1 import my_function, my_function_derivative
from common.utils import get_cheb_points, get_accuracy
from hermit_polynomial import get_hermite_polynomial
from terminaltables import AsciiTable
from typing import Callable
import numpy as np


def my_function_second_derivative(x):
    return np.sin(x / 2 + 0.2) / (2 * np.cos(x / 2 + 0.2) ** 3) - 2


def get_polynomial_accuracy(function: Callable, function_derivative: Callable,
                            interpolate_function: Callable,
                            interpolation_points_count=5,
                            get_points: Callable = np.linspace,
                            function_second_derivative: Callable = None):
    interval = (-2, 2)

    x = get_points(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x], [function_derivative(i) for i in x],
                            [function_second_derivative(i) for i in x]
                            if function_second_derivative is not None else None)

    polynomial = interpolate_function(*interpolation_points)
    polynomial_der = np.polyder(polynomial)
    polynomial_second_der = np.polyder(polynomial_der)

    polynomial_callable = lambda x: np.polyval(polynomial, x)
    polynomial_der_callable = lambda x: np.polyval(polynomial_der, x)
    polynomial_second_der_callable = lambda x: np.polyval(polynomial_second_der, x)

    return get_accuracy(function, polynomial_callable, interval), \
           get_accuracy(function_derivative, polynomial_der_callable, interval), \
           get_accuracy(function_second_derivative, polynomial_second_der_callable,
                        interval) if function_second_derivative is not None else None


def get_comparison_table(function, function_derivative, interpolating_function, function_second_derivative=None):
    table = [['n', 'P', 'Ch', 'P \'', 'Ch \'', 'P \'\'', 'Ch \'\'']]
    for n in range(3, 6):
        default_accuracy, default_der_accuracy, default_second_der_accuracy = get_polynomial_accuracy(function,
                                                                                                      function_derivative,
                                                                                                      interpolating_function,
                                                                                                      n,
                                                                                                      function_second_derivative=function_second_derivative)
        cheb_accuracy, cheb_der_accuracy, cheb_second_der_accuracy = get_polynomial_accuracy(function,
                                                                                             function_derivative,
                                                                                             interpolating_function, n,
                                                                                             get_cheb_points,
                                                                                             function_second_derivative=function_second_derivative)
        table.append(
            [n, default_accuracy, cheb_accuracy, default_der_accuracy, cheb_der_accuracy, default_second_der_accuracy,
             cheb_second_der_accuracy])

    return table


if __name__ == '__main__':
    table = get_comparison_table(my_function, my_function_derivative, get_hermite_polynomial,
                                 my_function_second_derivative)
    print(AsciiTable(table).table)
