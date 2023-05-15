import numpy as np
from typing import List


def __get_l_numerator(x: List[float], k: int, n: int):
    return np.poly([x[i] for i in range(n) if i != k])


def get_l(x: List[float], k: int, n: int):
    numerator = __get_l_numerator(x, k, n)
    return np.polydiv(numerator, np.polyval(numerator, x[k]))[0]


def get_lagrange_polynomial(x: List[float], y: List[float]):
    s = [0]
    n = len(x)
    for i in range(n):
        value = np.polymul(get_l(x, i, n), [y[i]])
        s = np.polyadd(value, s)
    return s


if __name__ == '__main__':
    x = [0, 2, 3]
    y = [1, 3, 2]
    print(get_lagrange_polynomial(x, y))
