import numpy as np
from lup_decomposition import lup_decomposition


def get_det(l_matrix: np.ndarray, u_matrix: np.ndarray, p_det: int) -> float:
    l_det = np.prod(np.diag(l_matrix))
    u_det = np.prod(np.diag(u_matrix))
    return l_det * u_det * p_det


if __name__ == '__main__':
    n = np.random.randint(2, 10)

    A = np.random.randint(-50, 50, size=(n, n)).astype(float)

    L, U, P, P_det = lup_decomposition(A.copy())

    A_det = get_det(L, U, P_det)

    print('det(A) =', A_det)
    print('det(A) == det(L) * det(U) * det(P):', np.isclose(np.linalg.det(A), A_det))
