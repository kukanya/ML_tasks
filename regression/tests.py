from regression.regression import *


def generate_noiseless_dataset(funcs, x_min, x_max, n):
    x = np.random.rand(n, 1) * (x_max - x_min) + x_min
    coeffs = (np.random.rand(len(funcs), 1) * 10 - 5)[:, 0]
    model = build_model(coeffs, funcs)
    y = []
    for x_i in x:
        n = list()
        for x_i_j in x_i:
            n.append(model(float(x_i_j)))
        y.append(n)
    y = np.array(y)
    return x, y, coeffs


def test_linear(left, right, n):
    x, y, coeffs = generate_noiseless_dataset(polynomials(1), left, right, n)
    model, model_coeffs = linear_regression(x, y, draw_flag=True)
    if np.allclose(coeffs, model_coeffs, rtol=1e-03):
        print("Coefficients are correct")


def test_polynomial(degree, left, right, n):
    x, y, coeffs = generate_noiseless_dataset(polynomials(degree), left, right, n)
    model, model_coeffs = polynomial_regression(x, y, degree, draw_flag=True)
    if np.allclose(coeffs, model_coeffs, rtol=1e-03):
        print("Coefficients are correct")


def test_nonlinear(funcs, left, right, n):
    x, y, coeffs = generate_noiseless_dataset(funcs, left, right, n)
    model, model_coeffs = nonlinear_regression(x, y, funcs, draw_flag=True)
    if np.allclose(coeffs, model_coeffs, rtol=1e-03):
        print("Coefficients are correct")
