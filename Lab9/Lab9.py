#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt


h = 1e-5
x_0 = 1 / np.sqrt(3)


def k(x):
    return np.exp(-x)


def q(x):
    if x < x_0:
        return x ** 3
    else:
        return x


def f(x):
    if x < x_0:
        return x * x - 1
    else:
        return 1


def TridiagonalMatrix(left_u, right_u, h):
    L = int(1 / h)

    
    l_a = int(np.floor(x_0 / h))
    l_b = l_a + 1

    x = np.zeros(L + 1)
    for l in range(0, L + 1):
        x[l] = l * h

    u = np.zeros(L + 1)
    u[0] = left_u
    u[L] = right_u

    a = np.zeros(L + 1)
    b = np.zeros(L + 1)
    c = np.zeros(L + 1)
    d = np.zeros(L + 1)
    alpha = np.zeros(L + 1)
    beta  = np.zeros(L + 1)


    # l in [1, l_a - 1]
    for l in range(1, l_a):
        a[l] = k((l + 0.5) * h)
        b[l] = -( k((l + 0.5) * h) + k((l - 0.5) * h) + q(l * h) * h * h )
        c[l] = k((l - 0.5) * h)
        d[l] = -f(l * h) * h * h


    # l in [l_b + 1, L - 1]
    for l in range(l_b + 1, L):
        a[l] = k((l + 0.5) * h)
        b[l] = -( k((l + 0.5) * h) + k((l - 0.5) * h) + q(l * h) * h * h )
        c[l] = k((l - 0.5) * h)
        d[l] = -f(l * h) * h * h


    alpha[1] = -a[1] / b[1]
    beta[1]  = (d[1] - c[1] * left_u) / b[1]

    alpha[L - 1] = -c[L - 1] / b[L - 1]
    beta[L - 1]  = (d[L - 1] - c[L - 1] * right_u) / b[L - 1]


    # l in [2, l_a - 1]
    for l in range(2, l_a):
        alpha[l] = -a[l] / (b[l] + c[l] * alpha[l - 1])
        beta[l]  = (d[l] - c[l] * beta[l - 1]) / (b[l] + c[l] * alpha[l - 1])


    # l in [L - 2, l_b + 1]
    for l in range(L - 2, l_b, -1):
        alpha[l] = -c[l] / (b[l] + a[l] * alpha[l + 1])
        beta[l]  = (d[l] - a[l] * beta[l + 1]) / (b[l] + a[l] * alpha[l + 1])


    u[l_a] = (k(l_a * h) * beta[l_a - 1] + k(l_b * h) * beta[l_b + 1]) / (k(l_a * h) * (1 - alpha[l_a - 1]) + k(l_b * h) * (1 - alpha[l_b + 1]))
    u[l_b] = u[l_a].copy()

    u[l_a - 1] = alpha[l_a - 1] * u[l_a] + beta[l_a - 1]
    u[l_b + 1] = alpha[l_b + 1] * u[l_b] + beta[l_b + 1]


    for l in range(l_a - 1, 0, -1):
        u[l] = alpha[l] * u[l + 1] + beta[l]

    for l in range(l_b + 1, L):
        u[l] = alpha[l] * u[l - 1] + beta[l]


    return x, u



def main():

    x, u = TridiagonalMatrix(2, 1, h)

    fig, ax = plt.subplots(figsize=[12, 8])

    plt.plot(x, u, color='red',  linewidth=3)

    plt.ticklabel_format(style='plain')

    plt.minorticks_on()
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.xlabel(r'$x$', fontsize=26)
    plt.ylabel(r'$u$', fontsize=26)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.plot([x_0, x_0], [0, 2], linewidth=2)

    plt.show()

    return

if __name__ == "__main__":
    main()