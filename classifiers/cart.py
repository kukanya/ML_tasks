import math

from metrics import gini, entropy
from utility.probability import ProbabilityCounter


class Condition(object):
    def __init__(self, i, p):
        self.i = i
        self.p = p
        self.cond = lambda x: x[self.i] < self.p

    def __str__(self):
        return "x[{}] < {}".format(self.i, self.p)

    def __call__(self, *args):
        return self.cond(*args)


class DecisionTree(object):
    def __init__(self, metric="entropy", min_samples_split=2):
        if metric == "gini":
            self.metric = gini
        elif metric == "entropy":
            self.metric = entropy
        else:
            self.metric = metric
        self.min_samples_split = min_samples_split

    def fit(self, X, Y, features=None):
        self.conditions = self._form_conditions(X, features)
        self.head = self._Node(self, X, Y)

    def predict(self, x):
        return self.head.predict(x).most_probable()

    def predict_many(self, X):
        return list(map(self.predict, X))

    def predict_probabilities(self, x):
        return self.head.predict(x).probabilities()

    @staticmethod
    def _form_conditions(X, cols=None):
        if cols is None:
            cols = range(X.shape[1])
        conditions = []
        for i in cols:
            col = X[:, i]
            points = list(set(col))
            points.sort()
            conditions.extend(list(map(lambda p: Condition(i, p), points)))
        return conditions

    @staticmethod
    def _split_by_condition(X, Y, cond):
        X_pos = []
        X_neg = []
        Y_pos = []
        Y_neg = []
        for (x, y) in zip(X, Y):
            if cond(x):
                X_pos.append(x)
                Y_pos.append(y)
            else:
                X_neg.append(x)
                Y_neg.append(y)
        return (X_pos, Y_pos), (X_neg, Y_neg)

    def _calculate_gains(self, X, Y):
        m = self.metric(Y)
        gains = []
        for cond in self.conditions:
            (_, Y_pos), (_, Y_neg) = self._split_by_condition(X, Y, cond)
            gains.append(m - (len(Y_pos) * self.metric(Y_pos) + len(Y_neg) * self.metric(Y_neg)) / len(Y))
        return gains

    class _Node(object):
        def __init__(self, tree, X, Y):
            gains = tree._calculate_gains(X, Y)
            max_gain = max(gains)
            if math.isclose(max_gain, 0, abs_tol=1e-5) or len(X) < tree.min_samples_split:
                self.is_leaf = True
                self.result = ProbabilityCounter(Y)
            else:
                self.is_leaf = False
                self.condition = list(filter(lambda p: p[1] == max_gain, zip(tree.conditions, gains)))[0][0]
                pos, neg = tree._split_by_condition(X, Y, self.condition)
                self.positive = tree._Node(tree, *pos)
                self.negative = tree._Node(tree, *neg)

        def predict(self, x):
            if self.is_leaf:
                return self.result
            else:
                if self.condition(x):
                    return self.positive.predict(x)
                else:
                    return self.negative.predict(x)
