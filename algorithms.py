import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import qr
from scipy.linalg import inv


def house(x):
    x = np.array(x.T)
    x = x / np.linalg.norm(x, np.inf)
    v = np.empty(x.size)
    v[1:] = x[1:]
    sigma = np.inner(x[1:], x[1:])
    if sigma < 1e-32:
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
    tol = 1e-32
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
    R, d = qr_householder(A.copy())
    solution = b.copy()
    m, n = A.shape
    for i in range(n):
        v = np.empty(m - i)
        v[0] = 1
        v[1:] = R[i + 1:, i].T
        v = np.matrix(v)
        solution[i:] -= np.array((d[i] * v.T * v * np.matrix(solution[i:]).T).T)[0]
    n = min(m, n)
    return backsub(R[:n, :n], solution[:n])

def householder2(a, b):
    A = a.copy()
    m, n = A.shape
    u = np.zeros((m, n))
    e1 = np.zeros((m, 1))
    e1[1] = 1
    for j in range(n):
        x = np.asmatrix(A[j:m, j]).T
        print("e1: ", e1[0:(m-j)].shape)
        print("x: ", x.shape)
        s = 1
        if x[1] <= 0:
            s = -1
        uj = x + s * np.linalg.norm(x) * e1[0:(m-j)]
        uj = uj / np.linalg.norm(uj)
        print("uj: ", uj.shape)
        print("A: ", A[j:m, j:n].shape)
        A[j:m, j:n] = A[j:m, j:n] - 2 * uj @ (uj.T @ A[j:m, j:n])
        u[j:m, j] = uj.transpose()

    y = np.asmatrix(b).T
    print("y: ", y.shape)
    for k in range(n):
        print("u: ", u[k:m, k:k+1].shape)
        y[k:m] = y[k:m] - 2 * u[k:m, k:k+1] @ (u[k:m, k:k+1].T @ y[k:m])

    n = min(m, n)
    # Aqui to usando a inversa, mas o resultado sai bem melhor
    # tentei usar o backsub, mas tah dando algo errado
    x =  inv(A[0:n, 0:n]) @ y[0:n]
    # x = np.squeeze(np.asarray(x))
    print("A: ", A[:n, :n].shape)
    print("A: ", A[:n, :n])
    # y = np.squeeze(np.asarray(y))
    # x = backsub(A[0:n, 0:n], y[0:n])

    x = np.squeeze(np.asarray(x))
    print("x: ", x.shape)

    return x



def scipy_qr(A, b):
    Q, R = qr(A)
    m, n = A.shape
    y = np.transpose(Q) @ b
    n = min(m, n)
    return backsub(R[:n, :n], y[:n])


def scipy_lstsq(A, b):
    p, res, rnk, s = lstsq(A, b)

    return p
