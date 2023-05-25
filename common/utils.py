import numpy as np
from random import randint
from typing import Tuple, List


def generate_matrix_with_eigenvalues(n: int) -> Tuple[np.ndarray, List[int]]:
    eigenvalues = [randint(-100, 100) for _ in range(n)]
    l = np.diag(eigenvalues).astype(float)
    c = np.random.randint(-100, 100, (n, n)).astype(float)
    a = np.linalg.inv(c) @ l @ c
    return a, eigenvalues


def __find__range_index(x: float, x_coords: List[float]):
    if x <= x_coords[0]:
        return 0
    n = len(x_coords)
    return next((i for i in range(1, n) if x_coords[i - 1] <= x < x_coords[i]), n - 1)


def get_cheb_points(start, end, num):
    return [np.cos(np.pi * (2 * k + 1) / (2 * (num + 1))) * (end - start) / 2 for k in range(num)]
