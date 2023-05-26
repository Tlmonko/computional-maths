import numpy as np
from task1 import my_function, get_approximation_polynomial
from common.draw_plot import draw_plot

if __name__ == '__main__':
    interval = (-1, 1)
    n = 5

    x_coords = np.linspace(*interval, n)
    y_coords = [my_function(x) for x in x_coords]

    moved_x = []
    moved_y = []
    for i in range(n):
        moved_x += [x_coords[i]] * 3
        moved_y += [y_coords[i] * 0.95, y_coords[i], y_coords[i] * 1.05]

    polynomial = get_approximation_polynomial(moved_x, moved_y)

    draw_plot(interval, [(moved_x, moved_y)], [my_function, lambda t: np.polyval(polynomial, t)],
              ['Primary function', 'approximation polynomial'])
