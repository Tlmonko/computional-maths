import numpy as np

from newton_default import newton
from newton_modified import modified_newton
from utils import timeit


@timeit
def combined_method(x, k):
    x, iterations_default, operations_default, _ = newton(x, k)
    x, iterations_modified, operations_modified, _ = modified_newton(x)
    return x, iterations_default + iterations_modified, operations_default + operations_modified


if __name__ == '__main__':
    x = np.array([[0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]]).transpose()
    k = int(input('k: '))
    _, iterations, operations, time = combined_method(x, k)

    print('Итераций:', iterations)
    print('Время:', time)
    print('Операций:', operations)
