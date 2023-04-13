import numpy as np
import math
import time

MAX_ITERATIONS = 5000


def timeit(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        return *result, time.time() - start

    return wrapper


def max_iterations_reached(iteration: int) -> bool:
    if iteration >= MAX_ITERATIONS:
        raise TimeoutError('Max iterations reached')
    return False


def F(x: np.array) -> np.array:
    return np.array([
        math.cos(x[1] * x[0]) - math.exp(-3 * x[2]) + x[3] * x[4] ** 2 - x[5] - math.sinh(2 * x[7]) * x[8] + 2 * x[
            9] + 2.000433974165385440,
        math.sin(x[1] * x[0]) + x[2] * x[8] * x[6] - math.exp(-x[9] + x[5]) + 3 * x[4] ** 2 - x[5] * (
                x[7] + 1) + 10.886272036407019994,
        x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904,
        2 * math.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - math.sin(x[1] ** 2) + math.cos(x[6] * x[9]) ** 2 - x[
            7] - 0.1707472705022304757,
        math.sin(x[4]) + 2 * x[7] * (x[2] + x[0]) - math.exp(-x[6] * (-x[9] + x[5])) + 2 * math.cos(x[1]) - 1.0 / (
                -x[8] + x[3]) - 0.3685896273101277862,
        math.exp(x[0] - x[3] - x[8]) + x[4] ** 2 / x[7] + math.cos(3 * x[9] * x[1]) / 2 - x[5] * x[
            2] + 2.0491086016771875115,
        x[1] ** 3 * x[6] - math.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) * math.cos(x[3]) + x[2] - 0.7380430076202798014,
        x[4] * (x[0] - 2 * x[5]) ** 2 - 2 * math.sin(-x[8] + x[2]) + 0.15e1 * x[3] - math.exp(
            x[1] * x[6] + x[9]) + 3.5668321989693809040,
        7 / x[5] + math.exp(x[4] + x[3]) - 2 * x[1] * x[7] * x[9] * x[6] + 3 * x[8] - 3 * x[0] - 8.4394734508383257499,
        x[9] * x[0] + x[8] * x[1] - x[7] * x[2] + math.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096])


def J(x: np.array) -> np.array:
    return np.array(
        [[-x[1] * math.sin(x[1] * x[0]), -x[0] * math.sin(x[1] * x[0]), 3 * math.exp(-3 * x[2]), x[4] ** 2,
          2 * x[3] * x[4],
          -1, 0, -2 * math.cosh(2 * x[7]) * x[8], -math.sinh(2 * x[7]), 2],
         [x[1] * math.cos(x[1] * x[0]), x[0] * math.cos(x[1] * x[0]), x[8] * x[6], 0, 6 * x[4],
          -math.exp(-x[9] + x[5]) - x[7] - 1, x[2] * x[8], -x[5], x[2] * x[6], math.exp(-x[9] + x[5])],
         [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
         [-x[4] / (x[2] + x[0]) ** 2, -2 * x[1] * math.cos(x[1] ** 2), -x[4] / (x[2] + x[0]) ** 2,
          -2 * math.sin(-x[8] + x[3]),
          1.0 / (x[2] + x[0]), 0, -2 * math.cos(x[6] * x[9]) * x[9] * math.sin(x[6] * x[9]), -1,
          2 * math.sin(-x[8] + x[3]), -2 * math.cos(x[6] * x[9]) * x[6] * math.sin(x[6] * x[9])],
         [2 * x[7], -2 * math.sin(x[1]), 2 * x[7], 1.0 / (-x[8] + x[3]) ** 2, math.cos(x[4]),
          x[6] * math.exp(-x[6] * (-x[9] + x[5])), -(x[9] - x[5]) * math.exp(-x[6] * (-x[9] + x[5])),
          2 * x[2] + 2 * x[0],
          -1.0 / (-x[8] + x[3]) ** 2, -x[6] * math.exp(-x[6] * (-x[9] + x[5]))],
         [math.exp(x[0] - x[3] - x[8]), -1.5 * x[9] * math.sin(3 * x[9] * x[1]), -x[5], -math.exp(x[0] - x[3] - x[8]),
          2 * x[4] / x[7], -x[2], 0, -x[4] ** 2 / x[7] ** 2, -math.exp(x[0] - x[3] - x[8]),
          -1.5 * x[1] * math.sin(3 * x[9] * x[1])],
         [math.cos(x[3]), 3 * x[1] ** 2 * x[6], 1, -(x[0] - x[5]) * math.sin(x[3]),
          x[9] / x[4] ** 2 * math.cos(x[9] / x[4] + x[7]),
          -math.cos(x[3]), x[1] ** 3, -math.cos(x[9] / x[4] + x[7]), 0, -1.0 / x[4] * math.cos(x[9] / x[4] + x[7])],
         [2 * x[4] * (x[0] - 2 * x[5]), -x[6] * math.exp(x[1] * x[6] + x[9]), -2 * math.cos(-x[8] + x[2]), 1.5,
          (x[0] - 2 * x[5]) ** 2, -4 * x[4] * (x[0] - 2 * x[5]), -x[1] * math.exp(x[1] * x[6] + x[9]), 0,
          2 * math.cos(-x[8] + x[2]),
          -math.exp(x[1] * x[6] + x[9])],
         [-3, -2 * x[7] * x[9] * x[6], 0, math.exp(x[4] + x[3]), math.exp(x[4] + x[3]),
          -7.0 / x[5] ** 2, -2 * x[1] * x[7] * x[9], -2 * x[1] * x[9] * x[6], 3, -2 * x[1] * x[7] * x[6]],
         [x[9], x[8], -x[7], math.cos(x[3] + x[4] + x[5]) * x[6], math.cos(x[3] + x[4] + x[5]) * x[6],
          math.cos(x[3] + x[4] + x[5]) * x[6], math.sin(x[3] + x[4] + x[5]), -x[2], x[1], x[0]]], dtype=float)
