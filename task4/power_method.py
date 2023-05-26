from common.utils import generate_matrix_with_eigenvalues
from typing import Union, List

import numpy as np


def power_method(matrix: Union[np.ndarray, np.matrix]) -> tuple[np.ndarray, List[float]]:
    y_vector = [1] * n
    z_vector = y_vector / np.linalg.norm(y_vector)
    prv_l = [float('+inf') for z in z_vector if not np.isclose(z, 0)]

    while True:
        y_vector = matrix @ z_vector
        l = [y / z for (y, z) in zip(y_vector, z_vector) if not np.isclose(z, 0)]
        z_vector = y_vector.copy() / np.linalg.norm(y_vector)
        if np.allclose(l, prv_l):
            return np.mean(l), z_vector
        prv_l = l.copy()


if __name__ == '__main__':
    n = int(input())
    A, eigenvalues, _ = generate_matrix_with_eigenvalues(n)

    l, x = power_method(A)
    print(f'Eigenvalue: {l}')
    print(f'Eigenvector: {x}')
