import numpy as np
from algorithms import householder


class Classifier:
    def __init__(self, n, labels, algorithm='householder'):
        """
        :param n: number of classes
        :param algorithm: LS algorithm used
        """
        if n < 3:
            print("Please enter a number of classes greater than 2.\n")
            return
        if len(labels) != n:
            print("Number of labels doesn't match with number of classes.\n")
            return
        if algorithm not in ('householder', 'givens', 'gram-schmidt'):
            print("Please enter a valid algorithm.\n")
            return
        self.n = n
        self.f = np.empty((n, n + 1))
        self.solve = householder

    def train(self, training_data):
        pass

    def evaluate(self):
        pass
