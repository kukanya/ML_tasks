import numpy as np
import bisect


class OneVsAll(object):
    def __init__(self, model_class, *margins):
        self.model_class = model_class
        self.margins = margins

    def fit(self, X, Y):
        self.unique_Y = sorted(list(set(Y)))
        self.classifiers = []
        self.classifiers_params = []
        filtered_X = X
        filtered_Y = Y
        for value in self.unique_Y[:len(self.unique_Y) - 1]:
            classifier = self.model_class(soft_margin=True, C=self.margins[value])
            classifier.fit(filtered_X, self.binarify_target(filtered_Y, value))
            self.classifiers.append(classifier.predict)
            self.classifiers_params.append((classifier.w, classifier.w0))
            filtered_X = np.array(
                    list(
                        map(lambda i: filtered_X[i], filter(lambda i: filtered_Y[i] != value, range(len(filtered_X))))))
            filtered_Y = np.array(list(filter(lambda y: y != value, filtered_Y)))

    def predict(self, X):
        # return np.array(list(map(lambda confs:
        #                          max(zip(confs, self.unique_Y), key=lambda z: z[0])[1],
        #                          np.array(list(map(lambda f: f(X), self.classifiers))).T)))
        # return np.array(list(map(lambda predicts: [i for (i, x) in enumerate(predicts) if x > 0]
        # with [i for (i, x) in enumerate(predicts) if x > 0] as l:
        #     l[0] if len(l) > 0 else len(predicts)
        # , np.array(list(map(lambda f: f(X), self.classifiers))).T)))
        classes = []
        for x in X:
            predicts = np.array(list(map(lambda f: f([x])[0], self.classifiers)))
            l = [i for (i, p) in enumerate(predicts) if p > 0]
            classes.append(self.unique_Y[l[0]] if len(l) > 0 else self.unique_Y[len(predicts)])
        return np.array(classes)

    @staticmethod
    def binarify_target(Y, value):
        bin_Y = np.zeros(Y.shape, dtype=np.int8)
        for i in range(len(Y)):
            bin_Y[i] = (-1, 1)[Y[i] == value]
        return bin_Y
