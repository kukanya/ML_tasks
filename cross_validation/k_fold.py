import numpy as np
from random import shuffle


def k_fold(k, x, y, error_func, model_builder, *args):
    ind = list(range(len(x)))
    shuffle(ind)
    cv = 0
    q = int(len(x) / k)
    r = len(x) % k
    nums = []
    for i in range(r):
        nums.append(i * (q + 1))
    for i in range(r, k):
        nums.append(i * q + r)
    nums.append(len(x))
    for i in range(k):
        outer_x = []
        new_x = []
        outer_y = []
        new_y = []
        for j in range(len(x)):
            if j in range(nums[i], nums[i + 1]):
                outer_x.append(x[ind[j]])
                outer_y.append(y[ind[j]])
            else:
                new_x.append(x[ind[j]])
                new_y.append(y[ind[j]])
        new_x = np.array(new_x)
        new_y = np.array(new_y)
        outer_x = np.array(outer_x)
        outer_y = np.array(outer_y)
        model = model_builder(new_x, new_y, *args)
        cv += error_func(model(outer_x), outer_y)
    cv /= k
    return cv
