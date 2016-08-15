import numpy as np


class KNN(object):
    def __init__(self, k, points, classes, dist_func, weighted=False):
        self.k = k
        self.points = points
        self.classes = classes
        self.dist_func = dist_func
        self.weighted = weighted

    def predict(self, x):
        distances = []
        for i in range(len(self.points)):
            distances.append((i, self.dist_func(self.points[i], x)))
        distances.sort(key=lambda d: d[1])
        class_counts = {}
        k = self.k
        while k > 0:
            for j in range(self.k):
                cl = self.classes[distances[j][0]]
                with np.errstate(divide='ignore'):
                    w = 1 / distances[j][1] if self.weighted else 1
                if cl in class_counts:
                    class_counts[cl] += w
                else:
                    class_counts[cl] = w
            max_freq = max(list(map(lambda cl: class_counts[cl], class_counts)))
            candidates = list(filter(lambda cl: class_counts[cl] == max_freq, class_counts))
            if len(candidates) == 1:
                break
            else:
                k -= 1
        return candidates[0]

    def predict_many(self, xs):
        classes = list(map(self.predict, xs))
        return classes
