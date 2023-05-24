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
