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


def calc_v(column, index) -> np.ndarray:
    s = calc_s(column, index)
    m = calc_m(s, column[index + 1])
    v = [0 if x <= index else column[x] for x in range(len(column))]
    v[index + 1] -= s
    return m * np.array([v])


def cast_to_hessenberg_form(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    eye = np.eye(n)
    for i in range(n - 1):
        v = calc_v(matrix[:, i], i)
        h = eye - 2 * v * np.transpose(v)
        matrix = h @ matrix
    return matrix.round(6)


def QR_shift(A, n):
    array = []
    current_bn = None
    shift_matr = copy.copy(A)
    while n > 0:
        I = np.eye(n)
        bn = shift_matr[n - 1][n - 1]
        if n > 1:
            bn_1 = shift_matr[n - 1][n - 2]
        else:
            bn_1 = None
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


def QR(A, n):
    bn = np.diag(A)
    matrix = A
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
            answer = np.diag(matrix)
            return answer

        array = []
        for i in range(n - 1):
            if abs(bn_1[i]) > delta:
                array.append(i)

        if len(array) == 1 and (False if previous_bn is None else (abs(bn - previous_bn) < abs(previous_bn / 3)).all()):
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
            while root_diff[0] > delta or root_diff[1] > delta:
                QQ, RR = np.linalg.qr(help_matrix)
                help_matrix = np.dot(RR, QQ)
                a = 1
                b = -help_matrix[0][0] - help_matrix[1][1]
                c = help_matrix[0][0] * help_matrix[1][1] - help_matrix[0][1] * help_matrix[1][0]
                D = np.complex_(b ** 2 - 4 * a * c + 0j)
                arr = []
                arr.append(np.complex_((-b + np.sqrt(D, dtype='complex_')) / 2))
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

        previous_bn = bn


def fix_A(diag, inv_C, C, n):
    fixA = np.eye(n)
    fixA = fixA * diag
    x = copy.copy(fixA[n - 1][n - 1])
    fixA[n - 1][n - 1] = fixA[n - 2][n - 2]
    fixA[n - 1][n - 2] = x
    fixA[n - 2][n - 1] = -x
    fixA = np.dot(inv_C, fixA)
    fixA = np.dot(fixA, C)
    return fixA, diag


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
    lambda_vec = QR_shift(hess, n)
    print(lambda_vec)
    print('=================================================================')
    for i in range(len(lambda_vec)):
        value, vector = inversed_power_method(a, lambda_vec[i])
        print(eigenvalues[i], "- текущее собственное число; ", "сдвиг:", lambda_vec[i],
              "cобственное число из алгоритма:",
              value,
              "; \nCобственный вектор из алгоритма:\n", vector)
    print('=================================================================')
    A_1, new_diag = fix_A(eigenvalues, np.linalg.inv(c), c, n)
    print("Новая матрица А:")
    print(A_1)
    print(QR(A_1, n))
