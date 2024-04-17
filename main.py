import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

def getF(x, M):
    size_x = len(x)
    F = np.zeros((size_x, M + 1))

    for i in range(size_x):
        for k in range(M + 1):
            F[i][k] = x[i] ** k

    return F

def getW(F, t):
    w = np.linalg.inv((F.T @ F)) @ F.T @ t
    #np.linalg.pinv(F) @ t
    return w

def getY(F, w):
    return F @ w

def getErrors(Y, t):
    result = (1 / 2) * sum((t - Y) ** 2)
    return result

errors = []

for M in range(1, 101):
    F = getF(x, M)
    w = getW(F, t)
    y = getY(F, w)

    E = getErrors(y, t)
    errors.append(E)

    if M == 1 or M == 8 or M == 100:
        plt.figure()
        plt.plot(x, z, label = 'z', color = 'navy')
        plt.scatter(x, t, label = 't', color = 'skyblue', s = 3)
        plt.plot(x, y, label = 'y', color = 'red')
        plt.title(f'M = {M}')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

plt.figure()
plt.plot(range(1, 101), errors)
plt.xlabel('M')
plt.ylabel('E')
plt.grid(True)
plt.show()
