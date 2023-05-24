import numpy as np
from lagrange_polynomial import get_lagrange_polynomial
from hermit_polynomial import get_hermite_polynomial
from common.draw_plot import draw_plot


def my_function(x):
    return np.tan(x / 2 + 0.2) - x ** 2 + 2


def my_function_derivative(x):
    return 1 / 2 * (np.cos(x / 2 + 0.2) ** 2) - 2 * x


def main():
    interval = (-2, 2)
    interpolation_points_count = 5

    x = np.linspace(*interval, num=interpolation_points_count)
    interpolation_points = (x, [my_function(i) for i in x])

    x_cheb = [np.cos(np.pi * (2 * k + 1) / (2 * (interpolation_points_count + 1))) * (interval[1] - interval[0]) / 2
              for k in range(interpolation_points_count)]
    cheb_interpolation_points = (x_cheb, [my_function(x) for x in x_cheb])

    lagrange_polynomial = get_lagrange_polynomial(*interpolation_points)
    lagrange_polynomial_cheb = get_lagrange_polynomial(*cheb_interpolation_points)

    hermite_polynomial = get_hermite_polynomial(x, [my_function(i) for i in x], [my_function_derivative(i) for i in x])

    draw_plot(interval, [interpolation_points, cheb_interpolation_points, interpolation_points, interpolation_points],
              [my_function, lambda x: np.polyval(lagrange_polynomial_cheb, x),
               lambda x: np.polyval(lagrange_polynomial, x),
               lambda x: np.polyval(hermite_polynomial, x)],
              ['Primary function', 'Lagrange', 'Lagrange chebyshev', 'Hermite'])


if __name__ == '__main__':
    main()
