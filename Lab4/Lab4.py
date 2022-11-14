#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt

x = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
y = [92228496, 106021537, 123202624,  132164569, 151325798, 179323175,  203211926, 226545805, 248709873, 281421906]


x1 = [0.0, 1.0, 2.0, 3.0, 4.0]
y1 = [0.00000, 0.50000, 0.86603, 1.00000, 0.86603]
x_star1 = 1.5

x2 = [0.0, 0.5, 0.9, 1.3, 1.7]
y2 = [-2.3026, -0.69315, -0.10536, 0.26236, 0.53063]
x_star2 = 0.8

x3 = [0.0, 1.7, 3.4, 5.1, 6.8]
y3 = [0.0000, 1.3038, 1.8439, 2.2583, 2.6077]
x_star3 = 3.0

x4 = [-0.4, -0.1, 0.2, 0.5, 0.8]
y4 = [1.9823, 1.6710, 1.3694, 1.0472, 0.6435]
x_star4 = 0.1

x5 = [0.0, 1.0, 2.0, 3.0, 4.0]
y5 = [1.0000, 1.5403, 1.5839, 2.0100, 3.3464]
x_star5 = 1.5


x_arr = [x1, x2, x3, x4, x5]
y_arr = [y1, y2, y3, y4, y5]
x_star_arr = [x_star1, x_star2, x_star3, x_star4, x_star5]



def DividedDiffs(x, y, k, j):

    if j == 0:
        return y[k]


    diff_1 = DividedDiffs(x, y, k + 1, j - 1)
    diff_2 = DividedDiffs(x, y, k, j - 1)

    return (diff_1 - diff_2) / (x[k + j] - x[k])


def CalculateAllDiffs(x, y):
    L = len(y)
    B = []

    for i in range(0, L):
        B.append(DividedDiffs(x, y, 0, i))

    return B


def NewtonPolynom(x, y):

    B = CalculateAllDiffs(x, y)
    
    def Polynom(s):
        Sum = 0
        Prod = 1

        for n in range(0, len(B)):
            Sum += B[n] * Prod
            Prod *= (s - x[n])

        return Sum
        
    return Polynom



def CalculateTriagCoeffsSpline(h, y):
    n = len(y)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    f = np.zeros(n)

    a[0] = 0
    b[0] = 1
    c[0] = 0
    f[0] = 0

    for k in range(1, n - 1):
        a[k] = h[k] / 6
        b[k] = (h[k] + h[k + 1]) / 3
        c[k] = h[k + 1] / 6
        f[k] = (y[k + 1] - y[k]) / h[k + 1] - (y[k] - y[k - 1]) / h[k]

    a[n - 1] = 0
    b[n - 1] = 1
    c[n - 1] = 0
    f[n - 1] = 0

    return a, b, c, f


def SolveTriagSystem(a, b, c, f):
    n = len(a)
    p = np.zeros(n)
    r = np.zeros(n)

    p[0] = c[0] / b[0]
    r[0] = f[0] / b[0]

    for k in range(1, n - 1):
        p[k] = c[k] / (b[k] - a[k] * p[k - 1])
        r[k] = (f[k] - a[k] * r[k - 1]) / (b[k] - a[k] * p[k - 1])

    for k in range(n - 2, -1, -1):
        c[k] = r[k] - p[k] * c[k + 1]

    return c


def Spline(x, y):
    n = len(x)
    h = np.zeros(n)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for k in range(1, len(h)):
        h[k] = x[k] - x[k - 1]


    a_triag, b_triag, c_triag, f_triag = CalculateTriagCoeffsSpline(h, y)
    c = SolveTriagSystem(a_triag, b_triag, c_triag, f_triag)


    for k in range(1, n):
        a[k] = y[k]
        d[k] = (c[k] - c[k - 1]) / h[k]
        b[k] = 1/6 * (2 * c[k] + c[k - 1]) * h[k] + (y[k] - y[k - 1]) / h[k]


    def SplineFunc(s):
        k = 0
        if s <= x[1]:
            k = 1
        elif s > x[n - 1]:
            k = n - 1
        else:
            for i in range(1, n):
                if s > x[i - 1] and s <= x[i]:
                    k = i
                    break

        return a[k] + b[k] * (s - x[k]) + 0.5 * c[k] * ((s - x[k]) ** 2) + 1/6 * d[k] * ((s - x[k]) ** 3)


    return SplineFunc
    


def main():

    showFirstGraphs = False

    for i in range(0, len(x_arr)):
        x_data = x_arr[i]
        y_data = y_arr[i]
        s = Spline(x_data, y_data)


        print('==================================================================')
        print('| x || ', end='')
        for elem in x_data:
            print('{:8.5f}  | '.format(elem), end='')


        print('\n==================================================================')
        print('| f || ', end='')
        for elem in y_data:
            print('{:8.5f}  | '.format(elem), end='')
        print('\n==================================================================')

        print(f'Interpolation at x* = {x_star_arr[i]}: f({x_star_arr[i]}) = {s(x_star_arr[i])}\n\n')

        if showFirstGraphs:
            x_s = np.arange(x_data[0], x_data[-1], 0.01)
            y_s = []
            for j in range(0, len(x_s)):
                y_s.append(s(x_s[j]))

            plt.plot(x_s, y_s, color='red', lw=2)
            plt.plot(x_data, y_data, 'o', color='black')
            plt.plot(x_star_arr[i], s(x_star_arr[i]), 'o', color='red')
            plt.minorticks_on()
            plt.grid(which='major', color='black', linestyle='-')
            plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)
            plt.show()


    print('------------------- USA population prediction --------------------')

    x_predict = 2010

    Poly = NewtonPolynom(x, y)    
    y_predict_poly = int(Poly(x_predict))
    print(f'Prediction by {x_predict} using Newton polynomial: {y_predict_poly}')

    x_poly = np.arange(1900, 2010, 0.01)
    y_poly = Poly(x_poly)


    MySpline = Spline(x, y)
    y_predict_spline = int(MySpline(x_predict))
    print(f'Prediction by {x_predict} using Spline interpolation: {y_predict_spline}')

    x_spline = np.arange(1900, 2010, 0.01)
    y_spline = []
    for i in range(0, len(x_spline)):
        y_spline.append(MySpline(x_spline[i]))


    fig, ax = plt.subplots(figsize=[9, 6])
    ax.set_title('USA population', fontsize=20)
    
    lineSpline, = ax.plot(x_spline, y_spline, color='red', lw=2, label='Spline interpolation')
    linePoly, = ax.plot(x_poly, y_poly, color='blue', lw=2, label='Newton polynomial')
    
    leg = ax.legend(fancybox=True, shadow=True)
    for line in leg.get_lines():
        line.set_linewidth(5.0)

    ax.plot(x, y, 'o', color='black')

    pointSpline, = ax.plot(x_predict, y_predict_spline, 'o', color='red')
    pointPoly,   = ax.plot(x_predict, y_predict_poly, 'o', color='blue')

    lines = [lineSpline, linePoly]
    lined = {}
    points = {lineSpline: pointSpline, linePoly: pointPoly}
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)
        lined[legline] = origline


    plt.xlabel('Year', fontsize=18)
    plt.ylabel('People', fontsize=18)
    plt.ticklabel_format(style='plain')
    plt.xticks(np.arange(1900, 2020, step=10))
    plt.yticks(rotation=30)
    plt.minorticks_on()
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)
    fig.tight_layout()
    

    def on_pick(event):
        legline = event.artist
        origline = lined[legline]
        point = points[origline]

        visible = not origline.get_visible()
        origline.set_visible(visible)
        point.set_visible(visible)

        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    return



if __name__ == "__main__":
    main()