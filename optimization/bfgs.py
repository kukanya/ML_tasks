import numpy as np

from optimization.base import generate_point, in_range
from optimization.dichotomy import dichotomy


def bfgs(lower, upper, func, grad, epsilon=0.01, save_path=False):
    norm = np.linalg.norm
    max_alpha = norm(upper - lower)
    I = np.identity(len(lower))
    C = I
    current_point = generate_point(lower, upper)
    if save_path:
        path = [current_point]
    while True:
        g = grad(current_point)
        p = np.inner(C, -g)
        p /= norm(p)
        while True:
            alpha, _ = dichotomy(np.array([0]), np.array([max_alpha]), lambda a: func(current_point + a * p),
                                 epsilon=epsilon * 0.01)
            next_point = current_point + alpha * p
            if in_range(next_point, lower, upper) or max_alpha < epsilon * 0.01:
                if save_path:
                    path.append(next_point)
                break
            else:
                max_alpha /= 2.0
        delta_x = next_point - current_point
        if norm(grad(next_point)) < epsilon or norm(delta_x) < epsilon * 0.01:
            if save_path:
                return next_point, func(next_point), path
            else:
                return next_point, func(next_point)
        else:
            delta_g = grad(next_point) - grad(current_point)
            beta = np.inner(delta_g, delta_x)
            C = (I - np.outer(delta_x, delta_g) / beta) * C * (I - np.outer(delta_x, delta_g) / beta) + \
                np.outer(delta_x, delta_x) / beta
            current_point = next_point

