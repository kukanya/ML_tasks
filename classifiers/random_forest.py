from random import choice

import numpy as np

from classifiers.cart import DecisionTree
from utility.probability import ProbabilityCounter


class RandomForest(object):
    def __init__(self, metric="entropy", n_trees=10, m_features=None):
        self.n_trees = n_trees
        self.m_features = m_features
        self.trees = [DecisionTree(metric) for _ in range(n_trees)]

    def fit(self, X, Y):
        for tree in self.trees:
            rand_X = []
            rand_Y = []
            ind = list(range(len(X)))
            for _ in range(len(X)):
                i = choice(ind)
                (x, y) = (X[i], Y[i])
                rand_X.append(x)
                rand_Y.append(y)
            rand_X = np.array(rand_X)
            rand_Y = np.array(rand_Y)
            if self.m_features is None or self.m_features == X.shape[1]:
                cols = None
            else:
                indices = list(range(X.shape[1]))
                cols = set()
                for _ in range(self.m_features):
                    i = choice(indices)
                    cols.add(i)
                    indices.remove(i)
            tree.fit(rand_X, rand_Y, cols)

    def predict(self, x):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x))
        return ProbabilityCounter(predictions).most_probable()

    def predict_many(self, X):
        return list(map(self.predict, X))