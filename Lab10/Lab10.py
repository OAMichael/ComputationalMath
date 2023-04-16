#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from sys import argv


h = 2.5
S_list = [78.2,   66.4,  43.0,  39.1,  33.2,  25.4,  31.3,  50.8,  87.7, 444.0, 523.2, 532.2, 538.5, 
          531.7, 527.2, 504.5, 498.7, 527.0, 570.6, 572.5, 566.7, 549.1, 535.4, 515.9, 486.6, 453.3,
          434.8, 420.0, 420.7, 437.2, 470.5, 480.8, 457.2, 408.4, 361.5, 340.0, 295.1, 257.9, 203.2,
          144.6, 103.6,  80.1,  64.5,  33.2,  21.5,  13.7,   7.8,  13.7,  23.4,  27.4,  23.4,  21.2, 
           18.5,  13.0,  11.3,   7.2,   5.8,   5.9,   9.8,   9.0,  19.5,  23.4,  37.1,  52.8,  86.0, 
          139.7, 139.7]


def S(x):
    return S_list[int(x/h)]


def S_prime(x):
    if int(x/h) >= len(S_list) - 1:
        return (S(x) - S(x - h)) / h

    if int(x/h) == 0:
        return (S(x + h) - S(x)) / h

    return (S(x + h) - S(x - h)) / (2*h)


def RungeKutta4Order(x_n, y_n, h, f):
    f_1 = f(x_n, y_n)
    f_2 = f(x_n + h / 2, y_n + f_1 * h / 2)
    f_3 = f(x_n + h / 2, y_n + f_2 * h / 2)
    f_4 = f(x_n + h, y_n + f_3 * h)

    # Means: y_n1 = y_{n + 1}

    y_n1 = y_n + h / 6 * (f_1 + 2 * f_2 + 2 * f_3 + f_4)

    return y_n1



def lambdaIteration(lam):
    F_lam = -1
    F_lam_prime = 0

    # psi = u, psi' = v
    def vecFuncUV(x, phaseVec):
        u = phaseVec[0][0]
        v = phaseVec[1][0]

        return np.array([[v], [-lam * lam * u - v * S_prime(x) / S(x)]])


    # du/d(lambda) = A, dv/d(lambda) = B
    def vecFuncABU(x, phaseVec):
        A = phaseVec[0][0]
        B = phaseVec[1][0]
        u = phaseVec[2][0]

        return np.array([[B], [-2 * lam * u - lam * lam * A - B * S_prime(x) / S(x)], [0]])


    vecUV  = np.array([[1], [0]])
    vecABU = np.array([[0], [0], [1]])

    x_list   = [0]
    UV_list  = [vecUV]
    ABU_list = [vecABU]


    for i in range(1, len(S_list)):
        x_list.append(i * h)

        new_UV = RungeKutta4Order(x_list[i - 1], UV_list[i - 1], h, vecFuncUV)
        UV_list.append(new_UV)

        new_ABU = RungeKutta4Order(x_list[i - 1], ABU_list[i - 1], h, vecFuncABU)
        ABU_list.append(new_ABU)
        ABU_list[i][2][0] = new_UV[0][0].copy()



    F_lam       = 8 * np.sqrt(S(x_list[-1])) * UV_list[-1][1][0]  + 3 * np.pi * np.sqrt(np.pi) * UV_list[-1][0][0]
    F_lam_prime = 8 * np.sqrt(S(x_list[-1])) * ABU_list[-1][1][0] + 3 * np.pi * np.sqrt(np.pi) * ABU_list[-1][0][0]

    return x_list, UV_list, F_lam, F_lam_prime


def ShootingMethod(init_lambda, epsilon):
    prev_lambda = init_lambda
    curr_lambda = init_lambda

    F_lam = -1

    UV_list = []


    while np.abs(F_lam) > epsilon:
        x_list, UV_list, F_lam, F_lam_prime = lambdaIteration(curr_lambda)
        curr_lambda = prev_lambda - F_lam / F_lam_prime
        prev_lambda = curr_lambda.copy()

    print(f'Valid lambda = {curr_lambda}')

    return x_list, UV_list, curr_lambda




def main():
    x_list, UV_list, valid_lambda = ShootingMethod(float(argv[1]), 1e-4)

    y_list = []
    for i in range(0, len(UV_list)):
        y_list.append(UV_list[i][0][0])
        
    fig, ax = plt.subplots(figsize=[12, 8])


    plt.plot(x_list, y_list, color='blue',  linewidth=2)

    plt.ticklabel_format(style='plain')

    plt.minorticks_on()
    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.xlabel(r'$x$' + ', мм', fontsize=20)
    plt.ylabel(r'$\Psi(x)$', fontsize=20)

    plt.xlim([0.0, 165.0])
    plt.ylim([-10.0, 15.0])

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)


    plt.show()
    return

if __name__ == "__main__":
    main()