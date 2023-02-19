#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt


def Deriv3Order(f, x, h):
    return (4 / 3 * (f(x + h) - f(x - h)) / 2 / h - 1 / 3 * (f(x + 2 * h) - f(x - 2 * h)) / 4 / h)


### -----------------------------------------------------------------------------------------------------------

### ------------------------- Section with functions to use Rosenbrock's method -------------------------------

### -----------------------------------------------------------------------------------------------------------


a = 0.435866521508459
beta_21 = a
beta_31 = a

beta    = a * (6 * a * a - 3 * a + 2) / (6 * a * a - 6 * a + 1)

beta_32 = beta - a
p_1     = a
p_3     = (6 * a * a - 6 * a + 1) / (6 * a * (beta - a))
p_2     = (1 - 2 * a - 2 * beta * p_3) / (2 * a)



def normVec(V):
    norm = 0.0
    for i in range(0, V.shape[0]):
        norm = max(norm, np.abs(float(V[i])))

    return norm


def UpperRelaxation(A, f, start, w=1.5, epsilon=1e-4):
    n = A.shape[0]
    
    D = np.zeros([n, n])
    L = np.zeros([n, n])
    U = np.zeros([n, n])

    prev_x = start
    curr_x = start

    for i in range(0, n):
        for j in range(0, n):
            if i > j:
                L[i, j] = A[i, j]
            elif i < j:
                U[i, j] = A[i, j]
            else:
                D[i, j] = A[i, j]

    ### This matrix is (D + wL)^-1
    M = np.linalg.inv(D + w * L)

    ### This matrix is (D + wL)^-1 * [(w-1)D + wU]
    V = np.matmul(M, (w - 1) * D + w * U)

    while normVec(f - np.matmul(A, curr_x)) >= epsilon:
        prev_x = curr_x
        curr_x = -np.matmul(V, prev_x) + np.matmul(w * M, f)

    return curr_x



def CalcMatrixForRosenbrock(x_n, y_n, h, f):
    Len = len(y_n)
    Jac = np.empty([Len, Len])
    for i in range(0, Len):
        for j in range(0, Len):
            def F_i(y):
                return f(x_n, y)[i, 0]


            def dFi_dyj(y):
                def F_i_of_y_j(y_j):
                    tmp = np.concatenate((y[:j], [[y_j]], y[(j+1):]))
                    return F_i(tmp)

                return Deriv3Order(F_i_of_y_j, y[j, 0], h)

            Jac[i, j] = dFi_dyj(y_n)

    return np.identity(Len) - a * h * Jac


def Rosenbrock(x_n, y_n, h, f):
    D_n = CalcMatrixForRosenbrock(x_n, y_n, h, f)

    b_1 = h * f(x_n, y_n)
    k_1 = UpperRelaxation(D_n, b_1, y_n)

    b_2 = h * f(x_n + beta_21 * h, y_n + beta_21 * k_1)
    k_2 = UpperRelaxation(D_n, b_2, k_1)

    b_3 = h * f(x_n + (beta_31 + beta_32) * h, y_n + beta_31 * k_1 + beta_32 * k_2)
    k_3 = UpperRelaxation(D_n, b_3, k_2)

    return y_n + p_1 * k_1 + p_2 * k_2 + p_3 * k_3





def Oregonator(independVar, phaseVec):
    x_component = phaseVec[0][0]
    y_component = phaseVec[1][0]
    z_component = phaseVec[2][0]

    new_x_component = 77.27 * (y_component + x_component * (1 - 8.375 * 1e-6 * x_component - y_component))
    new_y_component = 1 / 77.27 * (z_component - (1 + x_component) * y_component)
    new_z_component = 0.161 * (x_component - z_component)

    return np.array([[new_x_component], [new_y_component], [new_z_component]])



def main():
    start = np.array([[4], [1.1], [4]])
    h = 1e-2


    solution_points = [start]
    independ_points = [0.0]

    N = 80000
    for i in range(0, N):
        new_point = Rosenbrock(independ_points[i], solution_points[i], h, Oregonator)

        solution_points.append(new_point)
        independ_points.append(h * (i + 1))


    xs = []
    ys = []
    zs = []
    for elem in solution_points:
        xs.append(elem[0][0])
        ys.append(elem[1][0])
        zs.append(elem[2][0])

    fig, ax = plt.subplots(figsize=[12, 8])

    plt.plot(independ_points, xs, color='red',   linewidth=2, label='Вещество X')
    plt.plot(independ_points, ys, color='blue',  linewidth=2, label='Вещество Y')
    plt.plot(independ_points, zs, color='green', linewidth=2, label='Вещество Z')



    plt.title('Модель орегонатора', fontsize=24)
    plt.xlabel('Время, с', fontsize=20)
    plt.ylabel('Концентрация', fontsize=20)

    plt.xlim([0, 800])

    plt.ticklabel_format(style='plain')
    plt.xticks(np.arange(0, 801, step=100), fontsize=18)
    plt.yticks(fontsize=18)

    plt.minorticks_on()

    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.legend()

    plt.show()


    return

if __name__ == "__main__":
    main()