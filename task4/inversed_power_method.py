from typing import Union, Tuple

import numpy as np

from common.utils import generate_matrix_with_eigenvalues


def inversed_power_method(matrix: Union[np.ndarray, np.matrix], shift: float) -> Tuple[int, np.ndarray]:
    y_vector = [1] * matrix.shape[0]
    z_vector = y_vector / np.linalg.norm(y_vector)

    while True:
        y_vector = np.linalg.solve(matrix - shift * np.eye(matrix.shape[0]), z_vector)
        m = [z / y for z, y in zip(z_vector, y_vector) if not np.isclose(y, 0)]

        z_vector_prv = z_vector.copy()
        z_vector = y_vector / np.linalg.norm(y_vector)

        next_shift = shift + np.mean(m)
        if np.isclose(next_shift, shift) and np.isclose(np.linalg.norm(z_vector), np.linalg.norm(z_vector_prv)):
            return next_shift, z_vector
        shift = next_shift


if __name__ == '__main__':
    n = int(input())
    A, eigenvalues, _ = generate_matrix_with_eigenvalues(n)

    print('Given eigenvalue:', eigenvalues)

    l, x = inversed_power_method(A, eigenvalues[0] + 0.3)
    print(f'Calculated eigenvalue: {round(l, 6)}')
    print(f'Calculated eigenvector: {x}')
