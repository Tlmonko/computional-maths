import numpy as np

from newton_modified import modified_newton

if __name__ == '__main__':
    x = np.array([[0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]]).transpose()
    k = int(input('k: '))
    _, iterations, operations, time = modified_newton(x, k)

    print('Итераций:', iterations)
    print('Время:', time)
    print('Операций:', operations)
