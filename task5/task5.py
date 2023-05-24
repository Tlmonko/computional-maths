import numpy as np
from spline_10 import get_spline_10
from spline_31 import get_spline_31
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
    derivative_points = [my_function_derivative(i) for i in x]

    draw_plot(interval, [interpolation_points, interpolation_points, interpolation_points],
              [my_function, lambda t: get_spline_10(t, *interpolation_points),
               lambda t: get_spline_31(t, *interpolation_points, derivative_points)],
              ['function', 'spline 10', 'spline 31'])


if __name__ == '__main__':
    main()
