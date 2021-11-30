import numpy as np
from algorithms import householder
from algorithms import scipy_lstsq
from algorithms import scipy_qr


class Classifier:
    def __init__(self, labels, param_number, algorithm='householder'):
        """
        :param labels: label for each of the class
        :param param_number: number of parameters of classification
        :param algorithm: LS algorithm used
        """
        if len(labels) < 3:
            print("Please enter a number of classes greater than 2.\n")
            return
        if param_number < 1:
            print("Number of parameters must be grater than 0.\n")
            return
        if algorithm not in ('householder', 'givens', 'gram-schmidt', 'scipy_lstsq', 'scipy_qr'):
            print("Please enter a valid algorithm.\n")
            return
        if algorithm == 'householder':
            self.solve = householder
        elif algorithm == 'givens':
            pass
        elif algorithm == 'gram-schimidt':
            pass
        elif algorithm == 'scipy_lstsq':
            self.solve = scipy_lstsq
        elif algorithm == 'scipy_qr':
            self.solve = scipy_qr
        else:
            print("Please enter a valid algorithm.\n")
            return
        self.f = np.empty((len(labels), param_number + 1))
        self.labels = labels
        self.class_number = len(labels)
        self.param_number = param_number

    def train(self, training_data, training_targets):
        """
        :param training_data: numpy array of parameters
        :param training_targets: list of target labels
        """
        if training_data.size < 1 or len(training_targets) < 1:
            print("Data size must be greater than zero.\n")
            return
        if training_data.shape[0] != len(training_targets):
            print("Data size doesn't match result size.\n")
            return
        if training_data.shape[1] != self.param_number:
            print("Data size doesn't match parameter number.\n")
            return
        for training_result in training_targets:
            if training_result not in self.labels:
                print("There is a not specified class.\n")
                return
        data_size = len(training_targets)
        training_data = np.append(training_data, np.ones((data_size, 1)), axis=1)
        b = np.empty(data_size)
        for i in range(self.class_number):
            for j in range(data_size):
                if training_targets[j] == self.labels[i]:
                    b[j] = 1
                else:
                    b[j] = -1
            self.f[i] = self.solve(training_data, b)
        print(self.f)

    def evaluate(self, data):
        n_rows = data.shape[0]
        if n_rows < 1 or data.shape[1] != self.param_number:
            print("Data shape doesn't match.\n")
            return
        data = np.append(data, np.ones((n_rows, 1)), axis=1)
        result = []
        for row in data:
            result.append(self.labels[np.argmax(np.inner(self.f, row))])
        return result

    def change_algorithm(self, algorithm):
        if algorithm == 'householder':
            self.solve = householder
            return
        elif algorithm == 'givens':
            return
        elif algorithm == 'gram-schimidt':
            return
        elif algorithm == 'scipy_lstsq':
            self.solve = scipy_lstsq
            return
        elif algorithm == 'scipy_qr':
            self.solve = scipy_qr
            return
        print("Please enter a valid algorithm.\n")
