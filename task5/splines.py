from typing import List


def __get_fundamental_spline_1(x: float, x_coords: List[float], i: int):
    n = len(x_coords)
    if i == 0:
        return (x_coords[1] - x) / (x_coords[1] - x_coords[0])
    if i == n - 1:
        return (x - x_coords[-2]) / (x_coords[-1] - x_coords[-2])
    if x <= x_coords[i - 1] or x >= x_coords[i + 1]:
        return 0
    if x_coords[i - 1] <= x <= x_coords[i]:
        return (x - x_coords[i - 1]) / (x_coords[i] - x_coords[i - 1])
    # if x_coords[i] <= x <= x_coords[i + 1]:
    #     return (x_coords[i + 1] - x) / (x_coords[i + 1] - x_coords[i])


def __find__range_index(x: float, x_coords: List[float]):
    if x <= x_coords[0]:
        return 0
    n = len(x_coords)
    return next((i for i in range(1, n) if x_coords[i - 1] <= x < x_coords[i]), n - 1)


def get_spline_10(x, x_coords: List[float], y_coords: List[float]):
    i = __find__range_index(x, x_coords)
    if i == 0:
        y = y_coords[i]
        y_correction = 0
    else:
        y = y_coords[i] - y_coords[i - 1]
        y_correction = y_coords[i - 1]
    return y_correction + y * __get_fundamental_spline_1(x, x_coords, i)
