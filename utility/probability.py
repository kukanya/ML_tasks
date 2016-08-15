from collections import Counter


class ProbabilityCounter(Counter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probabilities(self):
        total = sum(self.values())
        return dict(zip(self.keys(), map(lambda v: v / total, self.values())))

    def most_probable(self):
        return self.most_common(1)[0][0]
