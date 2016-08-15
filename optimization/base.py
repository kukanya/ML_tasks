import numpy as np
from random import random


def generate_point(lower, upper):
    return np.array(tuple((l + random() * (u - l)) for l, u in zip(lower, upper)))


def in_range(point, lower, upper):
    res = all(tuple(l <= comp <= u for comp, l, u in zip(point, lower, upper)))
    return res
