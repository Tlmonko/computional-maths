import numpy as np
from task1 import draw_and_compare


def my_function(x):
    return abs(x) * (np.tan(x / 2 + 0.2) - x ** 2 + 2)


def my_function_derivative(x):
    return abs(x) * (1 / 2 * (np.cos(x / 2 + 0.2) ** 2) - 2 * x) + np.sign(x) * (np.tan(x / 2 + 0.2) - x ** 2 + 2)


if __name__ == '__main__':
    draw_and_compare(my_function, my_function_derivative)
