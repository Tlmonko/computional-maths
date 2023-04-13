import numpy as np

from newton_default import newton
from newton_modified import modified_newton

x = np.array([[0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5]]).transpose()

_, iterations_default, operations_default, _ = newton(x)
_, iterations_modified, operations_modified, _ = modified_newton(x)

print('Итераций (метод Ньютона):', iterations_default)
print('Операций (метод Ньютона):', operations_default)
print()
print('Итераций (модифицированный метод Ньютона):', iterations_modified)
print('Операций (модифицированный метод Ньютона):', operations_modified)
