from math import sqrt
from typing import Union, List
from common.utils import generate_matrix_with_eigenvalues
from inversed_power_method import inversed_power_method
import copy

import numpy as np


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def calc_s(column: Union[np.array, List[float]], index: int) -> float:
    return -sign(column[index + 1]) * np.sqrt(sum(column[x] ** 2 for x in range(index + 1, len(column))))


def calc_m(s, el) -> float:
    return 1 / sqrt(2 * s * (s - el))


def qr_with_shift(A, n):
    array = []
    current_bn = None
    shift_matr = copy.copy(A)
    while n > 0:
        I = np.eye(n)
        bn = shift_matr[n - 1][n - 1]
        bn_1 = shift_matr[n - 1][n - 2] if n > 1 else None
        shift_matr = shift_matr - bn * I
        Q, R = np.linalg.qr(shift_matr)
        shift_matr = np.dot(R, Q) + bn * I
        if (True if bn_1 is None else abs(bn_1) < eps) and (
                False if current_bn is None else abs(bn - current_bn) < abs(current_bn / 3)):
            array.append(bn)
            shift_matr = shift_matr[:n - 1, :n - 1]
            current_bn = None
            n -= 1
        else:
            current_bn = bn
    return array


def qr(matrix, n):
    bn = np.diag(matrix)
    matrix = matrix
    previous_bn = None
    while True:
        bn_1 = np.diagonal(matrix, offset=-1)
        Q, R = np.linalg.qr(matrix)
        matrix = np.dot(R, Q)
        flg1 = False
        flg2 = False
        for i in range(len(bn_1)):
            if abs(bn_1[i]) < delta:
                flg1 = True
            else:
                flg1 = False
                break
        if previous_bn is None:
            flg2 = False
        else:
            for i in range(len(bn_1)):
                if abs(bn[i] - previous_bn[i]) < abs(previous_bn[i] / 3):
                    flg2 = True
                else:
                    flg2 = False
                    break
        if flg1 and flg2:
            return np.diag(matrix)
        array = [i for i in range(n - 1) if abs(bn_1[i]) > delta]

        if len(array) == 1 and (False if previous_bn is None else (abs(bn - previous_bn) < abs(previous_bn / 3)).all()):
            return get_block(matrix, array)
        previous_bn = bn


def get_block(matrix, array):
    print("Матрица с блоком 2х2:")
    print(matrix)
    i = array[0]
    block = matrix[i:i + 2, i:i + 2]
    print("Блок 2х2:")
    print(block)

    help_matrix = block
    current_root = None
    previous_root = None
    root_diff = np.array([1e6, 1e6])
    a = 1
    while root_diff[0] > delta or root_diff[1] > delta:
        QQ, RR = np.linalg.qr(help_matrix)
        help_matrix = np.dot(RR, QQ)
        b = -help_matrix[0][0] - help_matrix[1][1]
        c = help_matrix[0][0] * help_matrix[1][1] - help_matrix[0][1] * help_matrix[1][0]
        D = np.complex_(b ** 2 - 4 * a * c + 0j)
        arr = [np.complex_((-b + np.sqrt(D, dtype='complex_')) / 2)]
        arr.append(np.complex_((-b - np.sqrt(D, dtype='complex_')) / 2))
        current_root = np.array(arr)
        if previous_root is not None:
            root_diff = np.abs(current_root - previous_root)
        previous_root = current_root
    print(current_root)
    answer = np.complex_(copy.copy(np.diag(matrix)))
    answer[i] = current_root[0]
    answer[i + 1] = current_root[1]
    return answer


def fix_matrix(diag, inv_c, c, n):
    fixA = np.eye(n)
    fixA = fixA * diag
    x = copy.copy(fixA[n - 1][n - 1])
    fixA[n - 1][n - 1] = fixA[n - 2][n - 2]
    fixA[n - 1][n - 2] = x
    fixA[n - 2][n - 1] = -x
    fixA = np.dot(inv_c, fixA)
    fixA = np.dot(fixA, c)
    return fixA, diag


def sgn_plus(i):
    return 1 if i >= 0 else -1


def calc_h(A, n, p, shift):
    I = np.eye(n)
    s = sum(A[i][p] ** 2 for i in range(p + shift, n))
    s = s ** 0.5
    s = -1 * sgn_plus(A[p + shift][p]) * s
    mu = 1 / abs(2 * s * (s - A[p + shift][p])) ** 0.5
    arr = []
    for k in range(n):
        if k < p + shift:
            arr.append(0)
        elif k == p + shift:
            arr.append(A[k][p] - s)
        else:
            arr.append(A[k][p])
    v = np.array(arr)
    v = mu * v
    v = v.reshape(n, 1)
    return I - 2 * np.dot(v, v.reshape(1, n))


def cast_to_hessenberg_form(matrix: np.ndarray):
    B = matrix.copy()
    for k in range(n - 2):
        h = calc_h(B, n, k, 1)
        C = np.dot(h, B)
        B = np.dot(C, h)
    return B


if __name__ == '__main__':
    eps = 1e-6
    delta = 1e-8

    n = int(input())
    a, eigenvalues, c = generate_matrix_with_eigenvalues(n)
    print(eigenvalues)
    # print(a)
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    hess = cast_to_hessenberg_form(a)
    print("QR - алгоритм со сдвигами:")
    lambda_vec = qr_with_shift(hess, n)
    print(lambda_vec)
    A_1, new_diag = fix_matrix(eigenvalues, np.linalg.inv(c), c, n)
    print("Новая матрица А:")
    print(A_1)
    print(qr(A_1, n))
