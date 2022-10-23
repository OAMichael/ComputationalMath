#!/usr/bin/python3

import numpy as np

""" 
For (x^2 - e^x / 5 = 0): 
    -One root on the [-1, 0] 
    -One root on the [ 0, 1]
    -One root on the [ 4, 5]
"""
def Function1(x):
    return (x ** 2) - np.exp(x) / 5

# For [-1, 0]
def Function1FPI(x):
    return -np.sqrt(np.exp(x) / 5)

# For [0, 1]
def Function2FPI(x):
    return np.sqrt(np.exp(x) / 5)

# For [4, 5]
def Function3FPI(x):
    return np.log(5 * (x ** 2))


# Defining all norms for our residual estimate to be correct
def normVec1(V):
    norm = 0.0
    for i in range(0, V.shape[0]):
        norm = max(norm, np.abs(float(V[i])))

    return norm


def normVec2(V):
    norm = 0.0
    for i in range(0, V.shape[0]):
        norm += np.abs(float(V[i]))

    return norm


def normVec3(V):
    return float(np.sqrt(np.dot(V.T, V)))


# Simple print function for displaying a matrix, including vector as matrix [N x 1] 
# The shape parameter is stands for displaying the shape of the matrix or vector
def printMatrix(M, shape=True):
    if shape:
        print(f"Matrix[{M.shape[0]}][{M.shape[1]}]")
    
    rows = M.shape[0]
    cols = M.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            start_symbol = ""
            end_symbol   = ""

            if j == 0:
                if i == 0:
                    start_symbol = "/ "
                elif i == rows - 1:
                    start_symbol = "\\ "
                else:
                    start_symbol = "| "
            
            if (j == cols - 1):
                if i == 0:
                    end_symbol = "  \\"
                elif i == rows - 1:
                    end_symbol = "  /"
                else:
                    end_symbol = "  |"

            print(start_symbol + "{:19.16f}".format(M[i, j]), end="")
            print(end_symbol + ("\n" if j == cols - 1 else "  "), end="")

    print("\n")



# Iterating function for Fixed Point Iteration method for solving nonlinear equations
# Takes start position, the method itself
# Following parameters are optional:
# - iterations: number of iterations to be performed in case of by iterations usage
# - epsilon: the precision to be achieved to stop the iterations in case of by epsilon usage
# - byIters: by default is True. If set, the method will be iterated <iterations> times and x will be returned. Otherwise, the method will be iterated until precision is achieved
# - func: function of our equation. Necessary for by epsilon usage 
def FixedPointIteration(start, method, iterations=100, epsilon=1e-6, byIters=True, func=None):
    x = start

    if byIters:
        for n in range(0, iterations):
            x = method(x)

        return x
    else:
        if func is None:
            print("For by epsilon usage you must provide original function")
            return None

        iters = 0
        while np.abs(func(x)) > epsilon:
            x = method(x)
            iters += 1

        return x, iters


# Defining all systems to solved by Newton's method
# We could hardcone only F's and calculate J's in runtime, but it requires to obtain the partial derivatives of F's which will add an extra error
def F1(x):
    F = np.array([[0.0], [0.0]])
    F[0] = np.sin(x[0] + 1.0) - x[1] - 1.2
    F[1] = 2 * x[0] + np.cos(x[1]) - 2.0

    return F


def J1(x):
    J = np.identity(2)
    J[0, 0] = np.cos(x[0] + 1.0)
    J[0, 1] = -1.0
    J[1, 0] = 2.0
    J[1, 1] = -np.sin(x[1])

    return J



def F2(x):
    F = np.array([[0.0], [0.0]])
    F[0] = 2 * (x[0] ** 2) - x[0] * x[1] - 5 * x[0] + 1
    F[1] = x[0] + 3 * np.log10(x[0]) - (x[1] ** 2)

    return F


def J2(x):
    J = np.identity(2)
    J[0, 0] = 4 * x[0] - x[1] - 5
    J[0, 1] = -x[0]
    J[1, 0] = 1 + 3 / (x[0] * np.log(10))
    J[1, 1] = -2 * x[1]

    return J


# Newton's method function. Takes start point of approximation, the system F itself, its Jacobian J
# Rest of parameters are the same as for FixedPointIteration() function, but norm is new one. Necessary to estimate the residual in case of by epsilon usage
def NewtonMethod(start, F, J, iterations=100, epsilon=1e-6, byIters=True, norm=normVec3):
    x = start

    if byIters:
        for n in range(0, iterations):
            x = x - np.matmul(np.linalg.inv(J(x)), F(x))

        return x
    else:
        iters = 0
        while norm(F(x)) > epsilon:
            x = x - np.matmul(np.linalg.inv(J(x)), F(x))
            iters += 1

        return x, iters



def main():
    """
        I've chosen this equation: x^2 - e^x / 5 = 0
        There are 3 roots:
            -One root on the [-1, 0] 
            -One root on the [ 0, 1]
            -One root on the [ 4, 5]

        We can compare two different approaches: by predefinec number of iteration to be performed and by redefined precision value
    """
    print("\033[96m\033[07m============================== FIXED POINT ITERATIONS FOR: x^2 - e^x / 5 = 0 ==================================\033[0m")
    print("\033[32m------------------------------------- BY PREDEFINED NUMBER OF ITERATIONS --------------------------------------\033[0m\n")
    
    iterations = 200
    print(f"Iterations: {iterations}")

    solution1 = FixedPointIteration(-1.0, Function1FPI, iterations)
    res = np.abs(Function1(solution1))
    print(f"Solution on the [-1, 0]: x = " + "{:19.16f}".format(solution1) + f" with residual r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    
    solution2 = FixedPointIteration(0.0,  Function2FPI, iterations)
    res = np.abs(Function1(solution2))
    print(f"Solution on the [0,  1]: x = " + "{:19.16f}".format(solution2) + f" with residual r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    
    solution3 = FixedPointIteration(4.0,  Function3FPI, iterations)
    res = np.abs(Function1(solution3))
    print(f"Solution on the [4,  5]: x = " + "{:19.16f}".format(solution3) + f" with residual r = {res}" + (" (< \u03b5ᶠ)\n" if res == 0.0 else "\n"))


    print("\033[32m------------------------------------------- BY PREDEFINED PRECISION -------------------------------------------\033[0m\n")
    
    epsilon = 1e-10
    print(f"Precision: \u03b5 = {epsilon}")

    solution1, iters1 = FixedPointIteration(-1.0, Function1FPI, epsilon=epsilon, byIters=False, func=Function1)
    res = np.abs(Function1(solution1))
    print( "Solution on the [-1, 0]: x = " + 
           "{:19.16f}".format(solution1) + 
           " with residual r = " + (" (< \u03b5ᶠ)" if res == 0.0 else "") + 
           "{:19.15e}".format(res) + 
          f" and {iters1} iterations")
    
    solution2, iters2 = FixedPointIteration(0.0,  Function2FPI, epsilon=epsilon, byIters=False, func=Function1)
    res = np.abs(Function1(solution2))
    print( "Solution on the [0,  1]: x = " + 
           "{:19.16f}".format(solution2) + 
           " with residual r = " + (" (< \u03b5ᶠ)" if res == 0.0 else "") + 
           "{:19.15e}".format(res) + 
          f" and {iters2} iterations" )
    
    solution3, iters3 = FixedPointIteration(4.0,  Function3FPI, epsilon=epsilon, byIters=False, func=Function1)
    res = np.abs(Function1(solution3))
    print( "Solution on the [4,  5]: x = " + 
           "{:19.16f}".format(solution3) + (" (< \u03b5ᶠ)" if res == 0.0 else "") + 
           " with residual r = " + 
           "{:19.15e}".format(res) + 
          f" and {iters3} iterations\n" )



    """
        Harnessing the Newton's method
        The chosen system is as follows:
            { sin(x + 1) - y = 1.2,
            { 2x + cos(y) = 2;

        As for nonlinear equation we can use two methods to obtain the solution: by number of iterations and by given precision
        The intersection somewhere at [0.4, 0.6]x[-0.3, -0.1]
        For predefined precision mode the residual is calculated only for one norm corresponding to the norm used in the mode to estimate the error: ||F(x)|| < epsilon 
    """
    print("\033[96m\033[07m================== NEWTON'S METHOD FOR: sin(x + 1) - y - 1.2 = 0, 2x + cos(y) - 2 = 0 =========================\033[0m")
    print("\033[32m------------------------------------- BY PREDEFINED NUMBER OF ITERATIONS --------------------------------------\033[0m\n")
    
    iterations = 20
    print(f"Iterations: {iterations}")

    start = np.array([[0.4], [-0.3]])
    solution4 = NewtonMethod(start, F1, J1, iterations)
    printMatrix(solution4, shape=False)

    res = normVec1(F1(solution4))
    print(f"Residual in 1st norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec2(F1(solution4))
    print(f"Residual in 2nd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec3(F1(solution4))
    print(f"Residual in 3rd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))


    print("\033[32m------------------------------------------- BY PREDEFINED PRECISION -------------------------------------------\033[0m\n")
    epsilon = 1e-10
    print(f"Precision: \u03b5 = {epsilon}")

    curr_norm = normVec2
    norm_num = "1st" if curr_norm is normVec1 else "2nd" if curr_norm is normVec2 else "3rd"

    solution4, iters4 = NewtonMethod(start, F1, J1, epsilon=epsilon, byIters=False, norm=curr_norm)
    printMatrix(solution4, shape=False)


    print(f"Iterations performed: {iters4}")
    res = curr_norm(F1(solution4))
    print(f"Residual in corresponding {norm_num} norm: r = {res}" + (" (< \u03b5ᶠ)\n" if res == 0.0 else "\n"))



    """
        The following lines the same as for the previous system. But now system is:
            { 2x^2 - xy - 5x + 1 = 0,
            { x + 3lg(x) - y^2 = 0;

        And main difference is existance of 2 intersections. That's why there are two block of code almost the same, they only differ by start approximation
        As mentioned before, there are two intersecions:
            -One somewhere at [1.4, 1.5]x[-1.5, -1.3]
            -One somewhere at [3.4, 3.5]x[ 2.2,  2.3]
    """

    # This block is for root at [1.4, 1.5]x[-1.5, -1.3]
    print("\033[96m\033[07m================== NEWTON'S METHOD FOR: 2x^2 - xy - 5x + 1 = 0, x + 3lg(x) - y^2 = 0 ==========================\033[0m")
    print("\033[92m####################################### ROOT AT [1.4, 1.5]x[-1.5, -1.3] #######################################\033[0m")

    print("\033[32m------------------------------------- BY PREDEFINED NUMBER OF ITERATIONS --------------------------------------\033[0m\n")
    
    iterations = 20
    print(f"Iterations: {iterations}")

    start = np.array([[1.4], [-1.5]])
    solution5 = NewtonMethod(start, F2, J2, iterations)
    printMatrix(solution5, shape=False)

    res = normVec1(F2(solution5))
    print(f"Residual in 1st norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec2(F2(solution5))
    print(f"Residual in 2nd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec3(F2(solution5))
    print(f"Residual in 3rd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))


    print("\033[32m------------------------------------------- BY PREDEFINED PRECISION -------------------------------------------\033[0m\n")
    epsilon = 1e-10
    print(f"Precision: \u03b5 = {epsilon}")

    curr_norm = normVec2
    norm_num = "1st" if curr_norm is normVec1 else "2nd" if curr_norm is normVec2 else "3rd"
    print(f"Used norm for the method: {norm_num}")

    solution5, iters5 = NewtonMethod(start, F2, J2, epsilon=epsilon, byIters=False, norm=curr_norm)
    printMatrix(solution5, shape=False)


    print(f"Iterations performed: {iters5}")
    res = curr_norm(F2(solution5))
    print(f"Residual in corresponding {norm_num} norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))



    print("\033[92m######################################## ROOT AT [3.4, 3.5]x[2.2, 2.3] ########################################\033[0m")
    print("\033[32m------------------------------------- BY PREDEFINED NUMBER OF ITERATIONS --------------------------------------\033[0m\n")
    
    iterations = 20
    print(f"Iterations: {iterations}")

    start = np.array([[3.5], [2.3]])
    solution5 = NewtonMethod(start, F2, J2, iterations)
    printMatrix(solution5, shape=False)

    res = normVec1(F2(solution5))
    print(f"Residual in 1st norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec2(F2(solution5))
    print(f"Residual in 2nd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))
    res = normVec3(F2(solution5))
    print(f"Residual in 3rd norm: r = {res}" + (" (< \u03b5ᶠ)" if res == 0.0 else ""))


    print("\033[32m------------------------------------------- BY PREDEFINED PRECISION -------------------------------------------\033[0m\n")
    epsilon = 1e-10
    print(f"Precision: \u03b5 = {epsilon}")

    curr_norm = normVec2
    norm_num = "1st" if curr_norm is normVec1 else "2nd" if curr_norm is normVec2 else "3rd"
    print(f"Used norm for the method: {norm_num}")

    solution5, iters5 = NewtonMethod(start, F2, J2, epsilon=epsilon, byIters=False, norm=curr_norm)
    printMatrix(solution5, shape=False)

    print(f"Iterations performed: {iters5}")
    res = curr_norm(F2(solution5))
    print(f"Residual in corresponding {norm_num} norm: r = {res}" + (" (< \u03b5ᶠ)\n" if res == 0.0 else "\n"))

    return



if __name__ == "__main__":
    main()