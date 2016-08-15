
from classifiers.svm.main import SVM
from classifiers.svm.one_vs_all import OneVsAll

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


def draw_points(X, Y, pred_Y):
    for i in range(len(X)):
        # if Y[i] != pred_Y[i]:
        #     print(X[i])
        shape = ['*', 'o', 's'][pred_Y[i]]
        color = ['r', 'g', 'b'][Y[i]]
        plt.plot(X[i][0], X[i][1], shape + color)
        # plt.annotate("%s" % i, xy=(X[i][0], X[i][1]))


def draw_line(w, w0, par, label):
    if abs(w[1]) > 1e-05:
        www = lambda x: (w0 - w[0] * x) / w[1]
        xx = np.arange(min(__X__[:, 0]) - 0.1, max(__X__[:, 0]) + 0.2, 0.1)
        plt.plot(xx, list(map(www, xx)), par, label=label)
        plt.plot(xx, list(map(lambda x: www(x) + (1.0 / w[1]), xx)),
                 par, label='SVM+')
        plt.plot(xx, list(map(lambda x: www(x) - (1.0 / w[1]), xx)),
                 par, label='SVM-')
    else:
        www = lambda x: (w0 / w[0])
        xx = np.arange(min(__X__[:, 1]) - 0.1, max(__X__[:, 1]) + 0.2, 0.1)
        plt.plot(list(map(www, xx)), xx, par, label=label)


iris = datasets.load_iris()
__X__ = iris.data[:, 2:]
__Y__ = iris.target
# __X__ = iris.data
# __Y__ = iris.target


model = OneVsAll(SVM, *(10, 0.05))
dump = model.fit(__X__, __Y__)
print(model.predict(__X__[:50]))
print(model.predict(__X__[50:100]))
print(model.predict(__X__[100:]))

# data = list(zip(__X__, __Y__))
# data0 = list(filter(lambda a: a[1] == 0, data))
# data1 = list(filter(lambda a: a[1] == 1, data))
# data2 = list(filter(lambda a: a[1] == 2, data))
#
# plt.plot(list(map(lambda a: a[0][0], data0)), list(map(lambda a: a[0][1], data0)), 'or', label='0 Class')
# plt.plot(list(map(lambda a: a[0][0], data1)), list(map(lambda a: a[0][1], data1)), 'og', label='1 Class')
# plt.plot(list(map(lambda a: a[0][0], data2)), list(map(lambda a: a[0][1], data2)), 'ob', label='2 Class')
# #
draw_points(__X__, __Y__, model.predict(__X__))

w, w0 = model.classifiers_params[0]
draw_line(w, w0, '-r', '0')

w, w0 = model.classifiers_params[1]
draw_line(w, w0, '-g', '1')

plt.xlim((min(__X__[:, 0]) - 0.1, max(__X__[:, 0]) + 0.1))
plt.ylim((min(__X__[:, 1]) - 0.1, max(__X__[:, 1]) + 0.1))


mistakes = [(x, y, y_pred) for (x, y, y_pred) in zip(__X__, __Y__, model.predict(__X__)) if y != y_pred]
print("mistakes:", len(mistakes))

plt.grid()
plt.show()


