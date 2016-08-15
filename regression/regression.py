from functools import reduce
import math
import numpy as np

from utility.visuals import draw_model_and_points


def polynomials(degree):
    result = [lambda x: 1.0]
    result += list(map(lambda k: lambda x: x ** (k + 1), range(degree)))
    return result


def trigonometrics(degree):
    result = [lambda x: 1.0]
    result += reduce(lambda l1, l2: l1 + l2,
                     map(lambda d: [lambda x: math.sin(d * x), lambda x: math.cos(d * x)],
                         range(1, degree + 1)))
    return result


def make_X_matrix(x, funcs):
    result = []
    for x_i in x[:, 0]:
        result.append(list(map(lambda f: f(x_i), funcs)))
    return np.matrix(result)


def evaluate_coeffs(X, y):
    return np.array(((X.T * X).I * X.T) * y)[:, 0]


def build_model(coeffs, funcs):
    return lambda x: sum(map(lambda a: a[0] * a[1](x), zip(coeffs, funcs)))


def linear_regression(x, y, draw_flag=False):
    funcs = polynomials(1)
    X = make_X_matrix(x, funcs)
    coeffs = evaluate_coeffs(X, y)
    model = build_model(coeffs, funcs)
    if draw_flag:
        draw_model_and_points(x, y, model)
    return model, coeffs


def polynomial_regression(x, y, degree, draw_flag=False):
    funcs = polynomials(degree)
    X = make_X_matrix(x, funcs)
    coeffs = evaluate_coeffs(X, y)
    model = build_model(coeffs, funcs)
    if draw_flag:
        draw_model_and_points(x, y, model)
    return model, coeffs


def nonlinear_regression(x, y, funcs, draw_flag=False):
    X = make_X_matrix(x, funcs)
    coeffs = evaluate_coeffs(X, y)
    model = build_model(coeffs, funcs)
    if draw_flag:
        draw_model_and_points(x, y, model)
    return model, coeffs
