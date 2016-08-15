from random import choice
import math
import numpy as np
import random as rnd
import scipy
from sklearn import datasets


class SVM(object):
    def __init__(self, soft_margin=False, C=None, K=None):
        self.soft_margin = soft_margin
        self.C = C
        self.K = K if K is not None else lambda x, y: np.dot(x, y)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.soft_margin:
            return self._fit_soft()
        else:
            self._fit_hard()

    def _fit_soft(self):
        l = len(self.Y)
        max_iterations = 100000
        steps = 0
        while True:
            solvable = True

            prev_w = np.zeros(self.X[0].shape)
            for i in range(len(prev_w)):
                prev_w[i] = float('nan')
            prev_w0 = float('nan')

            support_vectors_indices = self._init_support_vectors_indices()
            # bad seed
            # support_vectors_indices = [95, 23]
            # support_vectors_indices = [23, 95]
            # good seed
            # support_vectors_indices = [93, 41]

            # print(support_vectors_indices)
            # print([self.X[i] for i in support_vectors_indices])
            peripheral_vectors_indices = list(set(list(range(l))) - set(support_vectors_indices))
            violating_vectors_indices = []

            outer_flag = True
            while outer_flag and steps < max_iterations:

                inner_flag = True
                while inner_flag and steps < max_iterations:
                    # print(steps)
                    steps += 1
                    # print(support_vectors_indices)
                    s = len(support_vectors_indices)
                    try:
                        lams_eta = self._solve_quadratic_optimisation_task(*self._construct_optimisation_task_soft(
                                support_vectors_indices, violating_vectors_indices))
                    except ValueError:
                        solvable = False
                        break

                    lams = np.array(lams_eta)[:s].reshape((s,))
                    # print("lams", lams)

                    inner_flag = self._move_from_support_vectors(support_vectors_indices, lams,
                                                                 peripheral_vectors_indices,
                                                                 destination_set_type="peripheral")
                    if not inner_flag:
                        # print("whynot")
                        inner_flag = self._move_from_support_vectors(support_vectors_indices, lams,
                                                                     violating_vectors_indices,
                                                                     destination_set_type="violating")
                        # print("inner flag", inner_flag)
                if not solvable:
                    break

                w, w0 = self._evaluate_model_params(support_vectors_indices, lams)

                # print(support_vectors_indices)
                # print([self.X[i] for i in support_vectors_indices])

                # print(w, w0)
                # if np.isclose(w, prev_w).all() and np.isclose([w0], [prev_w0]):
                #     break

                peripheral_vectors_margins = self._evaluate_margins(peripheral_vectors_indices, w, w0)
                outer_flag = self._move_to_support_vectors(peripheral_vectors_indices, peripheral_vectors_margins,
                                                           support_vectors_indices, origin_set_type="peripheral")
                if not outer_flag:
                    violating_vectors_margins = self._evaluate_margins(violating_vectors_indices, w, w0)
                    outer_flag = self._move_to_support_vectors(violating_vectors_indices, violating_vectors_margins,
                                                               support_vectors_indices, origin_set_type="violating")
                prev_w = w
                prev_w0 = w0

            if solvable:
                # print(steps)
                print("steps:", steps)
                print("support:", support_vectors_indices)
                # print("peripheral:", peripheral_vectors_indices)
                # print("violating:", violating_vectors_indices)
                # print([self.X[i] for i in support_vectors_indices])
                # print([self.Y[i] for i in support_vectors_indices])
                self.w = w
                self.w0 = w0
                break

    def _fit_hard(self):
        l = len(self.Y)

        support_vectors_indices = self._init_support_vectors_indices()
        peripheral_vectors_indices = list(set(list(range(l))) - set(support_vectors_indices))

        outer_flag = True
        while outer_flag:
            inner_flag = True
            while inner_flag:
                # print(support_vectors_indices)
                s = len(support_vectors_indices)

                lams_eta = self._solve_quadratic_optimisation_task(
                        *self._construct_optimisation_task_hard(support_vectors_indices))
                lams = np.array(lams_eta)[:s].reshape((s,))

                inner_flag = self._move_from_support_vectors(support_vectors_indices, lams, peripheral_vectors_indices,
                                                             destination_set_type="peripheral")

            w, w0 = self._evaluate_model_params(support_vectors_indices, lams)

            margins = self._evaluate_margins(peripheral_vectors_indices, w, w0)
            outer_flag = self._move_to_support_vectors(peripheral_vectors_indices, margins, support_vectors_indices,
                                                       origin_set_type="peripheral")
        self.w = w
        self.w0 = w0

    def _move_from_support_vectors(self, support_vectors_indices, lams, destination_set_indices, destination_set_type):
        # print("--------")
        # print(destination_set_type)
        # print(support_vectors_indices)
        # print(lams)
        s = len(support_vectors_indices)
        if destination_set_type == "peripheral":
            cond = lambda l: l <= 0
        elif destination_set_type == "violating":
            # print('C=', self.C)
            cond = lambda l: l >= self.C
        else:
            raise ValueError("`destination_set_type` must be 'peripheral' or 'violating'")
        if s > 2:
            indices_to_move = []
            lams_to_move = []
            # index_to_move = None
            for i in range(s):
                # print(cond(lams[i]))
                if cond(lams[i]):
                    indices_to_move.append(support_vectors_indices[i])
                    # index_to_move = support_vectors_indices[i]
                    lams_to_move.append(lams[i])
                    # break
            while len(indices_to_move):
            # if index_to_move is not None:
            #     print("not None!")
                ind = choice(range(len(indices_to_move)))
                index_to_move = indices_to_move.pop(ind)
                lam = lams_to_move.pop(ind)
                y_to_move = self.Y[index_to_move]
                # print("Y TO MOVE",y_to_move)
                # print([self.Y[i] for i in support_vectors_indices])
                if len(list(filter(lambda i: self.Y[i] == y_to_move, support_vectors_indices))) > 1:
                    support_vectors_indices.remove(index_to_move)
                    destination_set_indices.append(index_to_move)
                    # print("v", index_to_move, "moved to", destination_set_type, "; lam =", lam)
                    # print("--------")
                    return True
        # print("--------")
        return False

    @staticmethod
    def _move_to_support_vectors(origin_set_indices, margins, support_vectors_indices, origin_set_type):
        if origin_set_type == "peripheral":
            cond = lambda m: m <= 1
        elif origin_set_type == "violating":
            cond = lambda m: m >= 1
        else:
            raise ValueError("`origin_set_type` must be 'peripheral' or 'violating'")
        indices_to_move = []
        margins_to_move = []
        for i in range(len(origin_set_indices)):
            if cond(margins[i]):
                indices_to_move.append(origin_set_indices[i])
                margins_to_move.append(margins[i])
        if len(indices_to_move):
            ind = choice(range(len(indices_to_move)))
            index_to_move = indices_to_move.pop(ind)
            m = margins_to_move.pop(ind)
            origin_set_indices.remove(index_to_move)
            support_vectors_indices.append(index_to_move)
            # print("v", index_to_move, "moved from", origin_set_type, "; m =", m)
            return True
        return False

    def _evaluate_margins(self, vectors_indices, w, w0):
        return [self.Y[i] * (np.dot(w, self.X[i]) - w0) for i in vectors_indices]

    def _evaluate_model_params(self, support_vectors_indices, lams):
        # print(lams)
        w = sum(map(lambda z: self.X[z[0]] * self.Y[z[0]] * z[1], zip(support_vectors_indices, lams)))
        w0 = np.median(np.array(list(map(lambda i: np.dot(w, self.X[i]) - self.Y[i], support_vectors_indices))))
        # print(w, w0)
        return w, w0

    def _construct_Q_matrix(self, vectors_indices1, vectors_indices2):
        k1 = len(vectors_indices1)
        k2 = len(vectors_indices2)
        Q = np.zeros((k1, k2))
        for i in range(k1):
            for j in range(k2):
                Q[i][j] = self.Y[vectors_indices1[i]] * self.Y[vectors_indices2[j]] * \
                          self.K(self.X[vectors_indices1[i]], self.X[vectors_indices2[j]])
        return np.matrix(Q)

    def _construct_optimisation_task_soft(self, support_vectors_indices, violating_vectors_indices):
        # print("constructing task for", support_vectors_indices)
        s = len(support_vectors_indices)
        c = len(violating_vectors_indices)
        if c:
            # print('c =', c)
            Q_SS = self._construct_Q_matrix(support_vectors_indices, support_vectors_indices)
            Q_SC = self._construct_Q_matrix(support_vectors_indices, violating_vectors_indices)
            g = self.C * Q_SC * np.matrix(np.ones((c, 1))) - np.matrix(np.ones((s, 1)))
            Y_S = np.matrix([[self.Y[i]] for i in support_vectors_indices])
            Y_C = np.matrix([[self.Y[i]] for i in violating_vectors_indices])
            f = -self.C * np.matrix(np.ones((c,))) * Y_C
            return Q_SS, g, Y_S.T, f
        else:
            return self._construct_optimisation_task_hard(support_vectors_indices)

    def _construct_optimisation_task_hard(self, support_vectors_indices):
        s = len(support_vectors_indices)
        Q_SS = self._construct_Q_matrix(support_vectors_indices, support_vectors_indices)
        g = np.matrix(-np.ones((s,))).T
        Y_S = np.matrix([[self.Y[i]] for i in support_vectors_indices])
        f = np.matrix(np.zeros((1,)))
        return Q_SS, g, Y_S.T, f

    @staticmethod
    def _solve_quadratic_optimisation_task(D, g, E, f):
        k = E.shape[0]
        A = np.matrix(np.vstack((np.hstack((D, E.T)), np.hstack((E, np.zeros((k, k)))))))
        b = np.matrix(np.vstack((-g, f)))
        if np.linalg.det(A) != 0:
            return SVM._solve_linear_system(A, b)
        else:
            tol = 1e-05
            while True:
                # print("решаю!")
                result, info = SVM._solve_linear_system_scipy(A, b, tol=tol)
                if info:
                    # print("info:", info)
                    tol *= 10
                    if tol >= 1:
                        # print("БИДА")
                        raise ValueError
                else:
                    return result

    @staticmethod
    def _solve_linear_system(A: np.matrix, b: np.matrix):
        # print("det A: ", np.linalg.det(A))
        return A.I * b

    @staticmethod
    def _solve_linear_system_scipy(A, b, tol=1e-05):
        return scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=1000)

    def predict_confidence(self, X):
        return np.array(list(map(lambda x: np.dot(self.w, x) - self.w0, X)))

    def predict_one(self, x):
        return np.sign(np.dot(self.w, x) - self.w0)

    def predict(self, X):
        return np.array(list(map(lambda x: self.predict_one(x), X)))

    def _init_support_vectors_indices(self):
        l = len(self.Y)
        i_prev = None
        i = rnd.choice(range(l))
        while True:
            R = []
            for j in range(l):
                if self.Y[j] != self.Y[i]:
                    R.append(np.linalg.norm(self.X[j] - self.X[i]))
                else:
                    R.append(math.inf)
            i_new = R.index(min(R))
            if i_new == i_prev:
                break
            else:
                i_prev = i
                i = i_new
        # return [11, 60]
        # return [5, 78]
        return [i, i_new]


def binarify_target(Y, value):
    bin_Y = np.zeros(Y.shape, dtype=np.int8)
    for i in range(len(Y)):
        bin_Y[i] = (-1, 1)[Y[i] == value]
    return bin_Y


if __name__ == '__main__':
    svm = SVM(soft_margin=True, C=50)
    # svm = SVM()
    iris = datasets.load_iris()
    __X__ = iris.data[:100, :2]
    __Y__ = iris.target[:100]
    Y_bin = binarify_target(__Y__, 1)
    # __X__ = np.array([[-1, 0], [0, 0], [0, 1], [1, 0], [1, 1], [2, 1]])
    # Y_bin = np.array([-1, -1, -1, 1, 1, 1])
    # __X__ = np.array([[-1, -2], [-1, 0], [0, -1], [1, 0], [1, 2], [0, 1]])
    # Y_bin = np.array([-1, -1, -1, 1, 1, 1])
    svm.fit(__X__, Y_bin)

    from matplotlib import pyplot as plt

    print(svm.w, svm.w0)
    data = list(zip(__X__, Y_bin))
    data1 = list(filter(lambda a: a[1] == 1, data))
    data_m_1 = list(filter(lambda a: a[1] == -1, data))
    plt.plot(list(map(lambda a: a[0][0], data1)), list(map(lambda a: a[0][1], data1)), 'or', label='1 Class')
    plt.plot(list(map(lambda a: a[0][0], data_m_1)), list(map(lambda a: a[0][1], data_m_1)), 'og', label='-1 Class')
    if abs(svm.w[1]) > 1e-05:
        www = lambda x: (svm.w0 - svm.w[0] * x) / svm.w[1]
        xx = np.arange(min(__X__[:, 0]) - 1, max(__X__[:, 0]) + 1, 0.1)
        plt.plot(xx, list(map(www, xx)), '-b', label='SVM')
        plt.plot(xx, list(map(lambda x: www(x) + (1.0 / svm.w[1]), xx)),
                 '-m', label='SVM+')
        plt.plot(xx, list(map(lambda x: www(x) - (1.0 / svm.w[1]), xx)),
                 '-y', label='SVM-')
    else:
        www = lambda x: (svm.w0 / svm.w[0])
        xx = np.arange(min(__X__[:, 0]) - 1, max(__X__[:, 0]) + 1, 0.1)
        plt.plot(list(map(www, xx)), xx, '-b', label='SVM')
        plt.plot(list(map(lambda y: www(y) + (1.0 / svm.w[0]), xx)), xx,
                 '-m', label='SVM+')
        plt.plot(list(map(lambda y: www(y) - (1.0 / svm.w[0]), xx)), xx,
                 '-y', label='SVM-')

    # plt.xticks([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    # plt.yticks([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    plt.legend(prop={'size': 6})
    # plt.legend()
    plt.grid()
    plt.show()
