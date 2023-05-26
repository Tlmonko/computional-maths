import numpy as np
from random import randint
from typing import Tuple, List, Callable


def generate_matrix_with_eigenvalues(n: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    eigenvalues = [randint(-10, 10) for _ in range(n)]
    l = np.diag(eigenvalues).astype(float)
    c = np.random.randint(-100, 100, (n, n)).astype(float)
    a = np.linalg.inv(c) @ l @ c
    return a, eigenvalues, c


def __find__range_index(x: float, x_coords: List[float]):
    if x <= x_coords[0]:
        return 0
    n = len(x_coords)
    return next((i for i in range(1, n) if x_coords[i - 1] <= x < x_coords[i]), n - 1)


def get_cheb_points(start, end, num):
    return [np.cos(np.pi * (2 * k + 1) / (2 * (num + 1))) * (end - start) / 2 for k in range(num + 1)]


def get_accuracy(function1: Callable, function2: Callable, interval: Tuple[int, int], accuracy_points_count=200):
    dx = (interval[1] - interval[0]) / accuracy_points_count
    return max(abs(function1(x) - function2(x)) * dx for x in np.linspace(*interval, num=accuracy_points_count))


def get_polynomial_accuracy(function: Callable, function_derivative: Callable,
                            interpolate_function: Callable,
                            interpolation_points_count=5,
                            get_points: Callable = np.linspace):
    interval = (-2, 2)

    x = get_points(*interval, num=interpolation_points_count)
    interpolation_points = (x, [function(i) for i in x], [function_derivative(i) for i in x])

    polynomial = interpolate_function(*interpolation_points)
    polynomial_der = np.polyder(polynomial)

    polynomial_callable = lambda x: np.polyval(polynomial, x)
    polynomial_der_callable = lambda x: np.polyval(polynomial_der, x)

    return get_accuracy(function, polynomial_callable, interval), get_accuracy(function_derivative,
                                                                               polynomial_der_callable, interval)
