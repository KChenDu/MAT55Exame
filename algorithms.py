import numpy as np


def house(x):
    x = np.array(x.T)
    x = x / np.linalg.norm(x, np.inf)
    v = np.empty(x.size)
    v[1:] = x[1:]
    sigma = np.inner(x[1:], x[1:])
    if sigma < 1e-12:
        beta = 0
    else:
        alpha = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = -sigma / (x[0] + alpha)
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)
        v /= v[0]
    return v, beta


def qr_householder(A):
    m, n = A.shape
    n = min(m, n)
    d = np.empty(n)
    for j in range(n):
        v, beta = house(A[j:, j])
        v = np.matrix(v)
        A[j:, j:] = (np.identity(m - j) - beta * v.T * v) * A[j:, j:]
        d[j] = beta
        A[j + 1:, j] = v[0, 1:]
    return A, d


def backsub(U, y):
    tol = 1e-12
    n = y.size
    for j in range(n - 1, 0, -1):
        if abs(U[j, j]) < tol:
            print("Singular matrix!\n")
        y[j] /= U[j, j]
        if j != 1:
            y[:j - 1] -= y[j] * np.array(U[:j - 1, j].T)[0]
        else:
            y[0] -= y[1] * U[0, 1]
    if abs(U[0, 0]) < tol:
        print("Singular matrix!\n")
    y[0] /= U[0, 0]
    return y


def householder(A, b):
    R, d = qr_householder(A[:, :])
    solution = b[:]
    m, n = A.shape
    for i in range(n):
        v = np.empty(m - i)
        v[0] = 1
        v[1:] = R[i + 1:, i].T
        v = np.matrix(v)
        solution[i:] -= np.array((d[i] * v.T * v * np.matrix(solution[i:]).T).T)[0]
    n = min(m, n)
    return backsub(R[:n, :n], solution[:n])
