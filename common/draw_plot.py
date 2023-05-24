from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import numpy as np


def draw_plot(interval: Tuple[float, float], points: List[Tuple], functions: List[Callable], legends: List[str] = None):
    x_coords = np.linspace(*interval, num=100)

    for function in functions:
        plt.plot(x_coords, [function(x) for x in x_coords])

    for points_plenty in points:
        plt.plot(*points_plenty, 'x')

    if legends is not None:
        plt.legend(legends)
    plt.show()
