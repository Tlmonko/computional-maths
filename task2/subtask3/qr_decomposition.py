import numpy as np
from typing import List, Tuple


def __get_normalized_vector(vector: np.array) -> np.array:
    return vector / np.linalg.norm(vector)


def __get_projection(vector: np.array, base: np.array) -> np.array:
    return (np.dot(vector, base) / np.dot(base, base)) * base


def gram_schmidt_process(vectors: List[np.array]) -> List[np.array]:
    result = []
    for index, vector in enumerate(vectors):
        result.append(vector - sum(__get_projection(vector, result[i]) for i in range(index)))
    return [__get_normalized_vector(vector) for vector in result]


def qr_decomposition(matrix: np.array) -> Tuple[np.array, np.array]:
    matrix = matrix.copy().astype(float)
    Q = np.array(gram_schmidt_process(matrix))
    R = np.dot(Q.T, matrix)
    return Q, R


matrix = np.array([np.array([1, 3, 4]), np.array([2, 3, 1]), np.array([4, 2, 3])])
print(qr_decomposition(matrix))
print('A == QR', np.allclose(matrix, np.dot(*qr_decomposition(matrix))))
