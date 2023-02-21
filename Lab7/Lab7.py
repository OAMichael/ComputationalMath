#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt


h = 5e-3


def Deriv3Order(f, x, h):
    return (4 / 3 * (f(x + h) - f(x - h)) / 2 / h - 1 / 3 * (f(x + 2 * h) - f(x - 2 * h)) / 4 / h)


def NewtonMethod(start, F, J, epsilon=1e-6, funcJacobian=True):
    curr_x = start
    prev_x = start

    while True:
        if funcJacobian:
            curr_x = curr_x - np.matmul(np.linalg.inv(J(curr_x)), F(curr_x))
        else:
            curr_x = curr_x - np.matmul(np.linalg.inv(J), F(curr_x))
        if normVec(curr_x - prev_x) < epsilon:
            break
        prev_x = curr_x

    return curr_x



### -----------------------------------------------------------------------------------------------------------

### -------------------------- Section with functions to use Radau's method ------------------------------------

### -----------------------------------------------------------------------------------------------------------

def CalcJacobianForRadau(F, y):
    TotalLen = len(y)
    J = np.empty([TotalLen, TotalLen])

    for i in range(0, TotalLen):
        for j in range(0, TotalLen):
            def F_i(y):
                return F(y)[i, 0]


            def dFi_dyj(y):
                def F_i_of_y_j(y_j):
                    tmp = np.concatenate((y[:j], [[y_j]], y[(j+1):]))
                    return F_i(tmp)

                return Deriv3Order(F_i_of_y_j, y[j, 0], h)

            J[i, j] = dFi_dyj(y)

    return J


def Radau3Order(x_n, y_n, h, f):
    def F(u):
        f_1 = np.array([[u[0, 0]], [u[1, 0]], [u[2, 0]]])
        f_2 = np.array([[u[3, 0]], [u[4, 0]], [u[5, 0]]])

        tmp1 = f_1 - f(x_n + h/3, y_n + h/12 * (5 * f_1 - f_2))
        tmp2 = f_2 - f(x_n + h,   y_n + h/4 * (3 * f_1 + f_2))

        return np.concatenate((tmp1, tmp2))


    start = np.concatenate((y_n, y_n))
    tmpJac = CalcJacobianForRadau(F, start)

    tmp = NewtonMethod(start, F, tmpJac, funcJacobian=False)


    f_1 = np.array([[tmp[0, 0]], [tmp[1, 0]], [tmp[2, 0]]])
    f_2 = np.array([[tmp[3, 0]], [tmp[4, 0]], [tmp[5, 0]]])

    return y_n + h/4 * (3 * f_1 + f_2)



### -----------------------------------------------------------------------------------------------------------

### ------------------------- Section with functions to use Rosenbrock's method -------------------------------

### -----------------------------------------------------------------------------------------------------------

a = 0.435866521508459
beta_21 = a
beta_31 = a

beta    = a * (6 * a * a - 3 * a + 2) / (6 * a * a - 6 * a + 1)

beta_32 = beta - a
p_1 = a
p_3 = (6 * a * a - 6 * a + 1) / (6 * a * (beta - a))
p_2 = (1 - 2 * a - 2 * beta * p_3) / (2 * a)



def normVec(V):
    norm = 0.0
    for i in range(0, V.shape[0]):
        norm = max(norm, np.abs(float(V[i])))

    return norm


def UpperRelaxation(A, f, start, w=1.5, epsilon=1e-6):
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


def Rosenbrock3Order(x_n, y_n, h, f):
    D_n = CalcMatrixForRosenbrock(x_n, y_n, h, f)

    b_1 = h * f(x_n, y_n)
    k_1 = UpperRelaxation(D_n, b_1, y_n)

    b_2 = h * f(x_n + beta_21 * h, y_n + beta_21 * k_1)
    k_2 = UpperRelaxation(D_n, b_2, k_1)

    b_3 = h * f(x_n + (beta_31 + beta_32) * h, y_n + beta_31 * k_1 + beta_32 * k_2)
    k_3 = UpperRelaxation(D_n, b_3, k_2)

    return y_n + p_1 * k_1 + p_2 * k_2 + p_3 * k_3



### -----------------------------------------------------------------------------------------------------------

### --------------------------- Section with functions to use Euler's method ----------------------------------

### -----------------------------------------------------------------------------------------------------------

def JacobianForImplicitEuler2Order(y):
    y1 = y[0, 0]
    y2 = y[1, 0]
    y3 = y[2, 0]

    Jacobian = np.identity(3)
    
    Jacobian[0, 0] = 77.27 - 2 * 77.27 * 8.375 * 1e-6 * y1 - 77.27 * y2
    Jacobian[0, 1] = 77.27 - 77.27 * y1
    Jacobian[0, 2] = 0

    Jacobian[1, 0] = -1/77.27 * y2
    Jacobian[1, 1] = -1/77.27 - 1/77.27 * y1
    Jacobian[1, 2] = 1/77.27

    Jacobian[2, 0] = 0.161
    Jacobian[2, 1] = 0
    Jacobian[2, 2] = -0.161

    Jacobian = -np.identity(3) + h * Jacobian

    return Jacobian



def ImplicitEuler2Order(x_n_plus_1, y_n, h, f):
    def F(y):
        return y_n - y + h * f(x_n_plus_1, y)

    return NewtonMethod(y_n, F, JacobianForImplicitEuler2Order)



### -----------------------------------------------------------------------------------------------------------

### -------------------------------------------- Main functions -----------------------------------------------

### -----------------------------------------------------------------------------------------------------------

def Oregonator(independVar, phaseVec):
    x_component = phaseVec[0][0]
    y_component = phaseVec[1][0]
    z_component = phaseVec[2][0]

    new_x_component = 77.27 * (y_component + x_component * (1 - 8.375 * 1e-6 * x_component - y_component))
    new_y_component = 1 / 77.27 * (z_component - (1 + x_component) * y_component)
    new_z_component = 0.161 * (x_component - z_component)

    return np.array([[new_x_component], [new_y_component], [new_z_component]])



def main():

    methods = ["ImplicitEuler2Order", "Radau3Order", "Rosenbrock3Order"]

    # Choose what you want
    method = methods[2]

    start = np.array([[4], [1.1], [4]])
    solution_points = [start]
    independ_points = [0.0]

    print(f"Using method: {method}")

    N = 80000
    for i in range(0, N):
        independ_points.append(h * (i + 1))

        if method == "ImplicitEuler2Order":
            new_point = ImplicitEuler2Order(independ_points[i + 1], solution_points[i], h, Oregonator)
        elif method == "Radau3Order":
            new_point = Radau3Order(independ_points[i], solution_points[i], h, Oregonator)
        elif method == "Rosenbrock3Order":
            new_point = Rosenbrock3Order(independ_points[i], solution_points[i], h, Oregonator)
        
        solution_points.append(new_point)
        
        if i % (N // 100) == 0:
            print(f"\r{1 + int(i / N * 100)}%", end='\r')

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


    plt.ticklabel_format(style='plain')

    plt.grid(which='major', color='black', linestyle='-')
    plt.grid(which='minor', color='0.65', linestyle='--', linewidth=0.1)

    plt.legend()

    plt.show()


    return

if __name__ == "__main__":
    main()