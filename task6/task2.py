import numpy as np
from task1 import get_approximation_polynomial, my_function
from common.draw_plot import draw_plot
from common.utils import get_accuracy
from task5.lagrange_polynomial import get_lagrange_polynomial

if __name__ == '__main__':
    interval = (-1, 1)
    n = 5

    x_coords = np.linspace(*interval, n)
    interpolation_points = (x_coords, [my_function(x) * np.random.randint(95, 105) / 100 for x in x_coords])

    polynomial = get_approximation_polynomial(*interpolation_points)
    lagrange_polynomial = get_lagrange_polynomial(*interpolation_points)
    draw_plot(interval,
              [interpolation_points],
              [my_function, lambda t: np.polyval(polynomial, t), lambda t: np.polyval(lagrange_polynomial, t)],
              ['Primary function', 'Approximation polynomial', 'Lagrange polynomial'])
    accuracy = get_accuracy(my_function, lambda t: np.polyval(polynomial, t), interval)
    lagrange_accuracy = get_accuracy(my_function, lambda t: np.polyval(lagrange_polynomial, t), interval)

    print(accuracy, lagrange_accuracy)
