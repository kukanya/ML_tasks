from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import optimization.funcs as f


def fun(x, y):
    return x ** 2 + y


def draw(lower, upper, func, func_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(lower[0], upper[0], 0.05)
    y = np.arange(lower[1], upper[1], 0.05)
    _x_, _y_ = np.meshgrid(x, y)

    z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(_x_), np.ravel(_y_))])
    _z_ = z.reshape(_x_.shape)

    surf = ax.plot_surface(_x_, _y_, _z_, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(*func_values)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == '__main__':
    __lower__ = np.array([-3, -3])
    __upper__ = np.array([3, 3])
    __func__ = f.tricky
    draw(__lower__, __upper__, __func__)
