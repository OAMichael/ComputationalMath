#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt

x_even = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
y_even = [1.000000, 0.989616, 0.958851, 0.908852, 0.841471, 0.759188, 0.664997, 0.562278, 0.454649]

x_odd = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
y_odd = [1.000000, 0.989616, 0.958851, 0.908852, 0.841471, 0.759188, 0.664997, 0.562278, 0.454649, 0.345810]

x = x_odd
y = y_odd

def Trapezoidal(x, y):
    h = np.zeros(len(x) - 1)
    for k in range(len(h)):
        h[k] = x[k + 1] - x[k]

    Sum = 0
    for k in range(len(h)):
        Sum += 0.5 * (y[k + 1] + y[k]) * h[k]

    return Sum

def Simpson(x, y, regular=False):
    # Case of evenly distributed points along X axis
    if regular:
        h = x[1] - x[0]
        n = len(x) - 1

        Sum = 0
        S_odd = 0
        S_even = 0
        for k in range(1, n//2 + 1):
            S_odd += y[2 * k - 1]
        for k in range(1, n//2):
            S_even += y[2 * k]

        Sum = h/3 * (y[0] + y[2 * (n//2)] + 4 * S_odd + 2 * S_even)

        if n % 2 == 1:
            Sum += 5/12 * h * y[n]
            Sum += 2/3 * h * y[n - 1]
            Sum -= 1/12 * h * y[n - 2]
        return Sum
    # Case of randomly distributed points along X axis
    else:
        n = len(x) - 1
        h = np.zeros(n)
        for k in range(0, n):
            h[k] = x[k + 1] - x[k]

        Sum = 0
        for k in range(1, n, 2):
            hph, hdh, hmh = h[k] + h[k - 1], h[k] / h[k - 1], h[k] * h[k - 1]
            Sum += (hph / 6) * ((2 - hdh) * y[k - 1] + (hph**2 / hmh) * y[k] + (2 - 1 / hdh) * y[k + 1])

        if n % 2 == 1:
            Sum += y[n] * (2 * h[n - 1] ** 2 + 3 * h[n - 2] * h[n - 1]) / (6 * (h[n - 2] + h[n - 1]))
            Sum += y[n - 1] * (h[n - 1] ** 2 + 3 * h[n - 1] * h[n - 2]) / (6 * h[n - 2])
            Sum -= (y[n - 2] * h[n - 1] ** 3) / (6 * h[n - 2] * (h[n - 2] + h[n - 1]))
        
        return Sum


def Richardson(x, y, method=Trapezoidal, r=2):
    p = 0
    if method == Trapezoidal:
        p = 2
    # (len(x) % 2 == 1) <=> (len(h) % 2 == 0)
    if len(x) % 2 == 1:
        r = 2
    else:
        r = 3 

    S_h = method(x, y)

    x_rh = []
    y_rh = []

    for k in range(0, len(x)):
        if k % r == 0:
            x_rh.append(x[k])
            y_rh.append(y[k])

    S_rh = method(x_rh, y_rh)

    S = (r**p * S_h - S_rh) / (r**p - 1)

    return S

def main():
    print(f"Integral value by Trapezoidal rule:                      {Trapezoidal(x, y)}")
    print(f"Integral value by Trapezoidal rule with Richardson rule: {Richardson(x, y)}")
    print(f"Integral value by Simpson's rule:                        {Simpson(x, y, regular=True)}")
    return

if __name__ == "__main__":
    main()