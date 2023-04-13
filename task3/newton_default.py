import numpy as np

from utils import J, F, timeit, max_iterations_reached


@timeit
def newton(x, k=None):
    iterations = 1
    e = 10 ** (-6)
    if k == 0:
        return x, 0, 0

    operations = 0
    n = F(x).shape[0]
    while not max_iterations_reached(iterations):
        x_old = x.copy()
        x -= np.linalg.inv(J(x)).dot(F(x))
        operations += n ** 3 + n ** 2
        iterations += 1
        if k and iterations == k:
            break
        if np.linalg.norm(x - x_old) < e:
            break
    return x, iterations, operations


if __name__ == '__main__':
    x = np.array([[0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]]).transpose()
    _, iterations, operations, time = newton(x)

    print('Итераций:', iterations)
    print('Время:', time)
    print('Операций:', operations)
