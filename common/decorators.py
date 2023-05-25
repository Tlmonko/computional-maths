import numpy as np


def differentiate_polynomial(f):
    def wrapper(*args, **kwargs):
        return np.polyder(f(*args, **kwargs))

    return wrapper
