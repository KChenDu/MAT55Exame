import numpy as np
from classifier import Classifier
from algorithms import householder

A = np.array([[0, 0.7, 1],
             [-0.7, 0, 0.7],
             [-1, -0.7, 0],
             [-0.7, -1.0, -0.7],
             [0, -0.7, -1.0],
             [0.7, 0, -0.7],
             [1, 0.7, 0],
             [0.7, 1, 0.7],
             [0, -0.7, 1],
             [0.7, 0, -0.7],
             [-1.0, 0.7, 0],
             [0.7, -1.0, 0.7],
             [0, 0.7, -1.0],
             [-0.7, 0, 0.7],
             [1.0, -0.7, 0],
             [-0.7, 1.0, -0.7]])

b = np.array([0.7, 0, -0.7, -1, -0.7, 0, 0.7, 1.0, 0, 0, 0, 0, 0, 0, 0, 0])

print(householder(A, b))
print(np.linalg.lstsq(A, b)[0])
