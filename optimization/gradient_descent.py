from functools import reduce
import numpy as np

from optimization.base import generate_point, in_range
from optimization.dichotomy import dichotomy


def constant_step(step):
    while True:
        yield step


def decreasing_step(initial, coefficient):
    step = initial
    while True:
        yield step
        step *= coefficient


def descent(lower, upper, func, grad, step_gen, *step_args, normalize_grad=False, epsilon=0.01,
            print_result=False, return_step_count=False):
    step = step_gen(*step_args)
    step_count = 1
    current_point = generate_point(lower, upper)
    if print_result:
        print("Start: (", reduce(lambda a, b: "{}, {}".format(a, b), current_point), ")", sep="")
    while True:
        lam = next(step)
        g = grad(current_point)
        if normalize_grad:
            g /= np.linalg.norm(g)
        step_count += 1
        while True:
            next_point = current_point - lam * g
            if in_range(next_point, lower, upper):
                break
            else:
                lam /= 2.0
        if step_count > 100000 or np.linalg.norm(current_point - next_point) < epsilon:
            if print_result:
                print("Steps done:", step_count)
                print("Result: f(", reduce(lambda a, b: "{}, {}".format(a, b), next_point), ") =  ",
                      func(next_point)[0], sep="")
            if return_step_count:
                return next_point, func(next_point), step_count
            else:
                return next_point, func(next_point)
        else:
            current_point = next_point


def descent_optimal_step(lower, upper, func, grad, normalize_grad=False, epsilon=0.01,
                         print_result=False, return_step_count=False):
    step_count = 1
    dich_step_count = 0
    current_point = generate_point(lower, upper)
    if print_result:
        print("Start: (", reduce(lambda a, b: "{}, {}".format(a, b), current_point), ")", sep="")
    lam_max = np.linalg.norm(upper - lower)
    while True:
        g = grad(current_point)
        if normalize_grad:
            g /= np.linalg.norm(g)
        step_count += 1
        while True:
            lam, dsc = dichotomy(np.array([0]), np.array([lam_max]), lambda lam_: func(current_point - lam_ * g),

                                 epsilon=epsilon * 0.1)
            next_point = current_point - lam * g
            if in_range(next_point, lower, upper):
                dich_step_count += dsc
                break
            else:
                lam_max /= 2.0
        if np.linalg.norm(current_point - next_point) < epsilon:
            if print_result:
                print("Steps done:", step_count)
                print("Result: f(", reduce(lambda a, b: "{}, {}".format(a, b), next_point), ") =  ",
                      func(next_point)[0], sep="")
            if return_step_count:
                return next_point, func(next_point), step_count, dich_step_count
            else:
                return next_point, func(next_point)
        else:
            current_point = next_point
