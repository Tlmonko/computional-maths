import numpy as np
from scipy.integrate import quad

A = 1.5
B = 2.3

ALPHA = 0.2
BETA = 0

M3 = 2262.135631823018


def power(number, p):
    return np.sign(number) * abs(number) ** p


def estimate_error(points):
    return M3 / 6 * quad(lambda x: abs(p(x) * w(x, points)), A, B)[0]


def f(x):
    return 2 * np.cos(3.5 * x) * np.exp(5 * x / 3) + 3 * np.sin(1.5 * x) * np.exp(-4 * x) + 3


def p(x):
    return (power(x - A, -ALPHA)) * (power(B - x, -BETA))


def w(x, points):
    mul = 1
    for xi in points:
        mul *= (x - xi)
    return mul


def calc_mu(power, start=A, end=B):
    return (end - A) ** (power + 1 - ALPHA) / (power + 1 - ALPHA) - (start - A) ** (power + 1 - ALPHA) / (
            power + 1 - ALPHA)


def calc_A(points):
    a = np.array([[(xi - A) ** s for xi in points] for s in range(len(points))])
    b = np.array([calc_mu(i) for i in range(len(points))])

    return np.linalg.solve(a, b)


if __name__ == '__main__':
    REAL_VALUE = 32.21951452884234295708696008290380201405

    points = [A, (A + B) / 2, B]
    a = calc_A(points)
    calculated_value = sum(f(x) * a[i] for i, x in enumerate(points))
    print('Calculated value', calculated_value)
    print('Estimated methodological error', estimate_error(points))
    print('Real error', abs(calculated_value - REAL_VALUE))
