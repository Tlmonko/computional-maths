import itertools
import numpy as np
from common.draw_plot import draw_plot
from common.utils import get_accuracy, get_cheb_points


def my_function(x):
    return x ** 2 * np.cos(x)


def get_approximation_polynomial(x, y):
    print(x, y)
    N = len(x)
    n = 3
    A = np.zeros((n + 1, n + 1))
    for k, j in itertools.product(range(n + 1), range(n + 1)):
        A[k][j] = sum(x[i] ** k * x[i] ** j for i in range(N))

    b = np.zeros(n + 1)
    for j in range(n + 1):
        b[j] = sum(y[i] * x[i] ** j for i in range(N))
    c = np.linalg.solve(A, b)
    return c[::-1]


if __name__ == '__main__':
    interval = (-1, 1)
    n = 5

    x_coords = np.linspace(*interval, n)
    interpolation_points = (x_coords, [my_function(x) for x in x_coords])

    x_cheb = get_cheb_points(*interval, n)
    cheb_interpolation_points = (x_cheb, [my_function(x) for x in x_cheb])

    polynomial = get_approximation_polynomial(*interpolation_points)
    polynomial_cheb = get_approximation_polynomial(*cheb_interpolation_points)

    draw_plot(interval, [interpolation_points, cheb_interpolation_points],
              [my_function, lambda t: np.polyval(polynomial, t), lambda t: np.polyval(polynomial_cheb, t)],
              ['Primary function', 'approximation', 'Chebyshev points approximation'])
    accuracy = get_accuracy(my_function, lambda t: np.polyval(polynomial, t), interval)
    accuracy_cheb = get_accuracy(my_function, lambda t: np.polyval(polynomial_cheb, t), interval)
    print(accuracy, accuracy_cheb)
