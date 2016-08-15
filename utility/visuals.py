from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from random import choice


def draw_points(x, y):
    plt.plot(x[:, 0], y[:, 0], "*r", label="Data points")
    plt.grid()
    plt.legend()
    plt.show()


def draw_model_and_points(x, y, model):
    grid = np.arange(min(x[:, 0]), max(x[:, 0]), 0.01)
    plt.plot(grid, list(map(model, grid)), '-g', label="Model")
    draw_points(x, y)


def draw_residuals(x, y, model):
    plt.plot(x, list(map(lambda pair: model(pair[0]) - pair[1], zip(x, y))), '*b', label="Residuals")
    plt.grid()
    plt.legend()
    plt.show()


def draw_levels_and_path(lower, upper, func, diapason, paths):
    fig, axs = plt.subplots(1)
    x = np.linspace(lower[0], upper[0], 100)
    y = np.linspace(lower[1], upper[1], 100)
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(*diapason, 100)
    Z = func((X, Y))[0]
    cs = axs.contourf(X, Y, Z, levels=levels, cmap=cm.gist_earth)
    fig.colorbar(cs, ax=axs, format="%.2f")
    for path in paths:
        colors = "bgrcym"
        symbols = "ops*^"
        x_plot = []
        y_plot = []
        for (x, y) in path:
            x_plot.append(x)
            y_plot.append(y)
        plt.plot(x_plot, y_plot, '-' + choice(symbols) + choice(colors))
    plt.grid()
    plt.show()
