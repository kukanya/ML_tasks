import math
import numpy as np


def polynomial(x):
    return x ** 3 - 4.0 * x ** 2 + 2.0 * x


def polynomial_grad(x):
    return 3 * x ** 2 - 8.0 * x + 2.0


def pseudo_rosenbrock(vec):
    (x, y) = vec
    return np.array([(1.0 - x ** 2) + 100.0 * (y - x ** 2) ** 2])


def pseudo_rosenbrock_grad(vec):
    (x, y) = vec
    return np.array([-2 * x * (1 + 200 * (y - x ** 2)), 200 * (y - x ** 2)])


def rosenbrock(vec):
    return np.array([sum(map(lambda a: (1.0 - a[0]) ** 2 + 100.0 * (a[1] - a[0] ** 2) ** 2, zip(vec, vec[1:])))])


def rosenbrock_grad(vec):
    n = len(vec)
    grad = [-2 * (1 - vec[0]) - 400 * vec[0] * (vec[1] - vec[0] ** 2)]
    for k in range(1, n - 1):
        grad.append(200 * (vec[k] - vec[k - 1] ** 2) - 2 * (1 - vec[k] - 400 * vec[k] * (vec[k + 1] - vec[k] ** 2)))
    grad.append(200 * (vec[n - 1] - vec[n - 2] ** 2))
    return np.array(grad)


def paraboloid(vec):
    (x, y) = vec
    return np.array([x ** 2 + y ** 2])


def paraboloid_grad(vec):
    (x, y) = vec
    return np.array([2 * x, 2 * y])


def tr(x):
    return 15.0 * math.exp(-(x ** 2)) * (x ** 2 - 0.4) + (0.4 ** 6) * (x ** 2) ** 3


def tr_grad(x):
    return (2 * x) * (15.0 * math.exp(-(x ** 2)) * (1.0 - (x ** 2 - 0.4)) +
                   3 * (0.4 ** 6) * (x ** 2))


def tricky(vec):
    (x, y) = vec
    return np.array([15.0 * math.e**(-(x ** 2 + y ** 2)) * (x ** 2 + y ** 2 - 0.4) + (0.4 ** 6) * (x ** 2 + y ** 2) ** 3])


def tricky_grad(vec):
    (x, y) = vec
    return np.array([
        (2 * x) * (15.0 * math.exp(-(x ** 2 + y ** 2)) * (1.0 - (x ** 2 + y ** 2 - 0.4)) +
                   3 * (0.4 ** 6) * (x ** 2 + y ** 2)**2),
        (2 * y) * (15.0 * math.exp(-(x ** 2 + y ** 2)) * (1.0 - (x ** 2 + y ** 2 - 0.4)) +
                   3 * (0.4 ** 6) * (x ** 2 + y ** 2)**2),
    ])