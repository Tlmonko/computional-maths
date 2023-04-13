from math import sqrt, atan, exp

from terminaltables import AsciiTable


def calc_u(x: float, accuracy: float) -> float:
    s = 0
    last_number = 1
    k = 0
    while abs(last_number) > accuracy:
        last_number = (-1) ** k * x ** (2 * k + 1) / (2 * k + 1)
        s += last_number
        k += 1
    return s


def calc_v(x: float, accuracy: float) -> float:
    s = 1
    last_number = 1
    k = 0
    while abs(last_number) > accuracy:
        k += 1
        last_number *= x / k
        s += last_number
    return s


def calc_w(x: float, accuracy: float) -> float:
    w = 1
    prv = 0
    while abs(w - prv) > accuracy:
        prv = w
        w = (w + (x + 1) / w) / 2
    return w


def z(x: float, u_accuracy, v_accuracy, w_accuracy) -> float:
    return calc_w(calc_u(0.8 * x + 0.2, u_accuracy), w_accuracy) / calc_v(2 * x + 1, v_accuracy)


def exact_z(x: float) -> float:
    return sqrt(1 + atan(0.8 * x + 0.2)) / exp(2 * x + 1)


Eu = (1e-6 / 1.53)
Ev = (1e-6 / 0.42)
Ew = (1e-6 / 2.73)

table = [['x', 'z(x)', 'z̃(x)', 'Δz(x)']]

for x in [x / 100 for x in range(10, 21)]:
    exact_value = exact_z(x)
    approximate_value = z(x, Eu, Ev, Ew)

    table.append([x, exact_value, approximate_value, abs(exact_value - approximate_value)])

print(AsciiTable(table).table)
