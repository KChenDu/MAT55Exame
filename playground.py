import numpy as np
from classifier import Classifier
from algorithms import householder
from algorithms import householder2

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

print(A.shape)
print(householder(A, b))
print(householder2(A, b))
# print(np.linalg.lstsq(A, b)[0])
