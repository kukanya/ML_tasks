import math
import numpy as np
from utility.probability import ProbabilityCounter


lp = lambda p: (lambda x1, x2: sum(map(lambda x: abs(x) ** p, x1 - x2)) ** (1 / p))

l2 = lambda x1, x2: np.linalg.norm(x1 - x2)
l1 = lambda x1, x2: sum(map(abs, x1 - x2))


precision = lambda y_model, y_true: \
    sum(list(map(lambda p: 1 if p[0] == p[1] else 0, zip(y_model, y_true)))) / len(y_model)

MSE = lambda y_model, y_true: sum(list(map(lambda p: (p[0] - p[1]) ** 2, zip(y_model, y_true)))) / len(y_model)


def gini(Y):
    if len(Y):
        probabilities = ProbabilityCounter(Y).probabilities()
        return 1 - sum(map(lambda k: probabilities[k] ** 2, probabilities))
    else:
        return -1


def entropy(Y):
    if len(Y):
        probabilities = ProbabilityCounter(Y).probabilities()
        return -sum(map(lambda k: probabilities[k] * math.log2(probabilities[k]), probabilities))
    else:
        return -1
