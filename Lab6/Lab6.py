#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt


def RungeKutta4Order(x_n, y_n, h, f):
    f_1 = f(x_n, y_n)
    f_2 = f(x_n + h / 2, y_n + f_1 * h / 2)
    f_3 = f(x_n + h / 2, y_n + f_2 * h / 2)
    f_4 = f(x_n + h, y_n + f_3 * h)

    # Means: y_n1 = y_{n + 1}

    y_n1 = y_n + h / 6 * (f_1 + 2 * f_2 + 2 * f_3 + f_4)

    return y_n1


def Adams4Order(x_prev4, y_prev4, h, f):

    # Means: x_n4 = x_{n - 4}
    x_n3 = x_prev4[0]
    x_n2 = x_prev4[1]
    x_n1 = x_prev4[2]
    x_n  = x_prev4[3]

    y_n3 = y_prev4[0]
    y_n2 = y_prev4[1]
    y_n1 = y_prev4[2]
    y_n  = y_prev4[3]

    y_n_plus_1 = y_n + h * (55/24 * f(x_n, y_n) - 59/24 * f(x_n1, y_n1) + 37/24 * f(x_n2, y_n2) - 3/8 * f(x_n3, y_n3))
    return y_n_plus_1



def Brusselator(independVar, phaseVec, A=1, B=1.5):
    x_component = phaseVec[0][0]
    y_component = phaseVec[1][0]

    new_x_component = A + x_component * x_component * y_component - (B + 1) * x_component
    new_y_component = B * x_component - x_component * x_component * y_component

    return np.array([[new_x_component], [new_y_component]])


def main():
    start = np.array([[1.0], [1.0]])
    h = 1e-3

    solution_points = [start]
    independ_points = [0.0]


    RungeKutta = False
    if RungeKutta:
        for i in range(0, 30000):
            new_point = RungeKutta4Order(independ_points[i], solution_points[i], h, Brusselator)

            solution_points.append(new_point)
            independ_points.append(h * (i + 1))

    else:
        for i in range(0, 30000):
            if i < 3:
                new_point = RungeKutta4Order(independ_points[i], solution_points[i], h, Brusselator)
            else:
                x_prev = [independ_points[i - 3], independ_points[i - 2], independ_points[i - 1], independ_points[i]]
                y_prev = [solution_points[i - 3], solution_points[i - 2], solution_points[i - 1], solution_points[i]]

                new_point = Adams4Order(x_prev, y_prev, h, Brusselator)

            solution_points.append(new_point)
            independ_points.append(h * (i + 1))

    xs = []
    ys = []
    for elem in solution_points:
        xs.append(elem[0][0])
        ys.append(elem[1][0])

    fig, ax = plt.subplots(figsize=[12, 8])

    plt.plot(independ_points, xs, color='red',  linewidth=2, label='Вещество X')
    plt.plot(independ_points, ys, color='blue', linewidth=2, label='Вещество Y')

    plt.title('Модель брюсселятора', fontsize=24)
    plt.xlabel('Время, с', fontsize=20)
    plt.ylabel('Концентрация', fontsize=20)

    plt.xlim([0, 30])
    plt.ylim([0, 2])

    plt.ticklabel_format(style='plain')
    plt.xticks(np.arange(0, 31, step=5), fontsize=18)
    plt.yticks(np.arange(0, 3, step=1), fontsize=18)

    plt.minorticks_on()
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.legend()

    plt.show()


    fig, ax = plt.subplots(figsize=[8, 8])

    plt.plot(xs, ys, color='black', linewidth=2)
    plt.ticklabel_format(style='plain')

    plt.title('Фазовый портрет', fontsize=24)
    plt.xlabel('Концентрация X', fontsize=20)
    plt.ylabel('Концентрация Y', fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
   
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)
    plt.show()

    return

if __name__ == "__main__":
    main()