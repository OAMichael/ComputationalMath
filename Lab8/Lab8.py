#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt


h = 1e-4


def RungeKutta4Order(x_n, y_n, h, f):
    f_1 = f(x_n, y_n)
    f_2 = f(x_n + h / 2, y_n + f_1 * h / 2)
    f_3 = f(x_n + h / 2, y_n + f_2 * h / 2)
    f_4 = f(x_n + h, y_n + f_3 * h)

    # Means: y_n1 = y_{n + 1}

    y_n1 = y_n + h / 6 * (f_1 + 2 * f_2 + 2 * f_3 + f_4)

    return y_n1



def alphaIteration(alpha):
    F_a = -1
    F_a_prime = 0

    def vecFuncYU(x, phaseVec):
        y = phaseVec[0][0]
        u = phaseVec[1][0]

        return np.array([[u], [x * np.sqrt(y)]])

    def vecFuncAB(x, phaseVec):
        A = phaseVec[0][0]
        B = phaseVec[1][0]
        y = phaseVec[2][0]

        return np.array([[B], [x / 2 / np.sqrt(y) * A], [0]])


    vecYU = np.array([[0], [alpha]])
    vecABY = np.array([[0], [1], [0.0001]])     # To avoid division by zero


    x_list   = [0]
    YU_list  = [vecYU]
    ABY_list = [vecABY]


    for i in range(1, int(1/h)):
        x_list.append(i * h)

        new_YU = RungeKutta4Order(x_list[i - 1], YU_list[i - 1], h, vecFuncYU)
        YU_list.append(new_YU)

        new_ABY = RungeKutta4Order(x_list[i - 1], ABY_list[i - 1], h, vecFuncAB)
        ABY_list.append(new_ABY)
        ABY_list[i][2][0] = new_YU[0][0].copy()

        F_a += new_YU[0][0] * h
        F_a_prime += new_ABY[0][0] * h



    return x_list, YU_list, F_a, F_a_prime



def ShootingMethod(init_alpha, epsilon):
    prev_alpha = init_alpha
    curr_alpha = init_alpha

    F_a = -1

    while np.abs(F_a) > epsilon:
        x_list, YU_list, F_a, F_a_prime = alphaIteration(curr_alpha)

        curr_alpha = prev_alpha - F_a / F_a_prime
        prev_alpha = curr_alpha.copy()

        print(f'Current error dF = {np.abs(F_a)}')

    print(f'Valid alpha = {curr_alpha}')

    return x_list, YU_list, F_a





def main():
    x_list, YU_list, integral = ShootingMethod(2.0, 1e-6)
    
    y_list = []
    for i in range(0, len(YU_list)):
        y_list.append(YU_list[i][0][0])


    integral += 1
    print(f'Intergal = {integral}')


    fig, ax = plt.subplots(figsize=[12, 8])

    plt.plot(x_list, y_list, color='red',  linewidth=2)

    plt.ticklabel_format(style='plain')

    plt.minorticks_on()
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$y$', fontsize=20)


    plt.show()

    return

if __name__ == "__main__":
    main()