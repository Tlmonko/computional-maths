import numpy as np

from task2.subtask1.linear_system import solve_linear_system_with_knowing_lup
from task2.subtask1.lup_decomposition import lup_decomposition
from utils import J, F, timeit, max_iterations_reached


@timeit
def modified_newton(x, k=None):
    iterations = 0
    e = 10 ** (-6)
    operations = 0
    n = J(x).shape[0]
    L, U, P, _ = lup_decomposition(J(x))
    operations += n ** 3

    while not max_iterations_reached(iterations):
        x_old = x.copy()
        delta = np.array([[*solve_linear_system_with_knowing_lup(L, U, P, -F(x))]])
        operations += 2 * n ** 2 + n
        x += delta.transpose()
        iterations += 1
        if np.linalg.norm(x - x_old) < e:
            break
        if k and iterations % k == 0:
            L, U, P, _ = lup_decomposition(J(x))
            operations += n ** 3
    return x, iterations, operations


if __name__ == '__main__':
    x = np.array([[0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]]).transpose()
    _, iterations, operations, time = modified_newton(x)

    print('Итераций:', iterations)
    print('Время:', time)
    print('Операций:', operations)
