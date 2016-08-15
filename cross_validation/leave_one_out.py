import numpy as np


def leave_one_out(x, y, error_func, model_builder, *args):
    cv = 0
    for i in range(len(x)):
        outer_x = np.array([x[i]])
        outer_y = np.array([y[i]])
        mask = np.ones(len(x), dtype=bool)
        mask[i] = 0
        new_x = x[mask]
        new_y = y[mask]
        model = model_builder(new_x, new_y, *args)
        cv += error_func(model(outer_x), outer_y)
    cv /= len(x)
    return cv
