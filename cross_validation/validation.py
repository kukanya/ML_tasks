def validate(X, Y, error_func, model):
    return error_func(model(X), Y)
