from optimization.base import generate_point


def monte_carlo(lower, upper, func, tries=10000):
    arg_min = generate_point(lower, upper)
    _min = func(arg_min)
    for i in range(tries - 1):
        arg = generate_point(lower, upper)
        val = func(arg)
        if val < _min:
            arg_min = arg
            _min = val
    return arg_min


def monte_carlo_hybrid(opt_method, *args, tries=100, **kwargs):
    arg_min, _min = opt_method(*args, **kwargs)
    for i in range(tries - 1):
        point, value = opt_method(*args, **kwargs)
        if value < _min:
            arg_min, _min = point, value
    return arg_min, _min
