from random import random


def dichotomy(left, right, func, epsilon=0.001):
    step_count = 0
    while right - left >= 2 * epsilon:
        step_count += 1
        mid = (left + right) / 2.0
        delta = random() * (right - left) / 2.0
        new_left = mid - delta
        new_right = mid + delta
        if func(new_left) > func(new_right):
            left = new_left
        else:
            right = new_right
    return (right + left) / 2, step_count
