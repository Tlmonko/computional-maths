from math import sqrt
from typing import Union, List

import numpy as np

from common.utils import generate_matrix_with_eigenvalues


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def calc_s(column: Union[np.array, List[float]], index: int) -> float:
    return -sign(column[index + 1]) * np.sqrt(sum(column[x] ** 2 for x in range(index + 1, len(column))))


def calc_m(s, el) -> float:
    return 1 / sqrt(2 * s * (s - el))


def calc_v(column, index) -> np.ndarray:
    s = calc_s(column, index)
    m = calc_m(s, column[index + 1])
    print(s)
    print(m)
    v = [0 if x <= index else column[x] for x in range(len(column))]
    v[index + 1] -= s
    print(v)
    return m * np.array([v])


def cast_to_hessenberg_form(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    eye = np.eye(n)
    for i in range(n - 1):
        v = calc_v(matrix[:, i], i)
        h = eye - 2 * v * np.transpose(v)
        print(h)
        matrix = h @ matrix
    return matrix.round(6)


if __name__ == '__main__':
    # n = int(input())
    # a, eigenvalues = generate_matrix_with_eigenvalues(n)
    # print(eigenvalues)
    # print(a)
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(cast_to_hessenberg_form(a))
