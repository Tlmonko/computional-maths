import numpy as np

from inverse import inverse


def get_inf_norm(matrix: np.ndarray) -> float:
    return np.max(np.sum(np.abs(matrix), axis=1))


def get_conditionality_number(matrix: np.ndarray) -> float:
    return get_inf_norm(matrix) * get_inf_norm(inverse(matrix))


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    A = np.random.randint(-50, 50, size=(n, n)).astype(float)

    print('A:', A, sep='\n', end='\n\n')

    print('cond(A):', get_conditionality_number(A))
