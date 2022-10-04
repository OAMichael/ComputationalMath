#!/usr/bin/python3

from math import factorial
import numpy as np
import itertools


n = 10
f = np.array([[1 / i] for i in range(1, n + 1)])
A = np.identity(n)

for i in range(0, n):
    for j in range(0, n):
        if i == j:
            A[i, j] = 1
        else:
            A[i, j] = 1 / (i + j + 2) # Need that extra +2 because i and j start with 0, but not with 1 


def printMatrix(M):
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

            print(start_symbol + "{:9.6f}".format(M[i, j]), end="")
            print(end_symbol + ("\n" if j == cols - 1 else "  "), end="")

    print("\n")



def printVector(V):
    print("[ ", end="")
    for i in range(0, V.shape[0]):
        print("{:6f}".format(float(V[i])), end="")
        print(" ]\n" if i == V.shape[0] - 1 else "  ", end="")  

    print("\n")


def normalize(V):
    norm = normVec3(V)
    if norm == 0:
        return np.array([[0] for i in range(0, V.shape[0])])
    else:
        return V / norm



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



def normMat1(A):
    n = A.shape[0]
    norm = 0.0
    for i in range(0, n):
        Sum = 0.0
        for j in range(0, n):
            Sum += np.abs(A[i, j])
        norm = max(Sum, norm)

    return norm



def normMat2(A):
    n = A.shape[0]
    norm = 0.0
    for j in range(0, n):
        Sum = 0.0
        for i in range(0, n):
            Sum += np.abs(A[i, j])
        norm = max(Sum, norm)

    return norm


def normMat3(A):
    return np.sqrt(findMaxEigenvalue(np.matmul(A.T, A)))


def IsLUCompatible(A):
    result = True
    
    n = A.shape[0]
    init_array = np.arange(0, n)

    ### Test for all main minors of original matrix to check if it is compatible for LU decomposition
    for minor_order in range(1, n + 1):
        # Calculate all possible combinations of numbers 0...n with size of minor order
        permutations = list(itertools.combinations(init_array, minor_order))

        for perm in permutations:
            M = A[[ [int(j)] for j in perm ], [int(j) for j in perm]]
            if np.linalg.det(M) == 0:
                result = False
                break

    return result



def IsPositiveDefinite(A):
    result = True
    for i in range(0, A.shape[0]):
        M = A[:A.shape[0] - i, :A.shape[1] - i]
        if np.linalg.det(M) == 0:
            result = False

    return result



def IsCholeskyCompatible(A):
    result = IsLUCompatible(A)
    if result:
        result = (A.all() == A.T.all()) and IsPositiveDefinite(A)

    return result
    


def findL(A):
    L = np.zeros([A.shape[0], A.shape[1]])

    for j in range(0, A.shape[0]):
        LSum = 0.0
        for k in range(0, j):
            LSum += L[j, k] * L[j, k]

        L[j, j] = np.sqrt(A[j, j] - LSum)

        for i in range(j + 1, A.shape[1]):
            LSum = 0.0
            for k in range(0, j):
                LSum += L[i, k] * L[j, k]
            L[i][j] = (1.0 / L[j, j] * (A[i, j] - LSum))


    ### Check if everything alright by comparing L with library result
    if np.linalg.cholesky(A).all() == L.all():
        print("Cholesky matrix L has been found successfully!\n\n")
    else:
        print("Failed to find Cholesky matrix!\n\n")
    
    return L


def solveForUpperTr(LT, f):
    n = f.shape[0]
    x = np.zeros(n)
    x[n - 1] = f[n - 1] / LT[n - 1, n - 1]

    # From back to begining
    for i in range(f.shape[0] - 2, -1, -1):
        Sum = 0.0
        for j in range(i, n):
            Sum += LT[i, j] * x[j]
        
        x[i] = (f[i] - Sum) / LT[i, i]

    return x.reshape([n, 1])



def solveForLowerTr(L, f):
    n = f.shape[0]
    x = np.zeros(n)
    x[0] = f[0] / L[0, 0]

    for i in range(1, n):
        Sum = 0.0
        for j in range(0, i):
            Sum += L[i, j] * x[j]
        
        x[i] = (f[i] - Sum) / L[i, i]

    return x.reshape([n, 1])



def findMaxEigenvalue(A, init_vector=np.array([[i + 1] for i in range(0, A.shape[0])])):
    iterations  = 500
    prev_vector = init_vector
    curr_vector = prev_vector

    for i in range(0, iterations):
        curr_vector = normalize(curr_vector)
        prev_vector = normalize(prev_vector)

        prev_vector = curr_vector
        curr_vector = np.matmul(A, curr_vector)

    maxEigenVal = np.max(np.abs(curr_vector / prev_vector))
    return maxEigenVal


def findMinEigenvalue(A, init_vector=np.array([[i + 1] for i in range(0, A.shape[0])])):
    return 1 / findMaxEigenvalue(np.linalg.inv(A), init_vector)



def IsUpperRelaxationCompatible(A):
    return IsPositiveDefinite(A)



def UpperRelaxation(A, f, w=1.5, epsilon=1e-6, init_vector=np.array([[0] for i in range(0, A.shape[0])]), norm=normVec1):
    n = A.shape[0]
    
    D = np.zeros([n, n])
    L = np.zeros([n, n])
    U = np.zeros([n, n])

    prev_x = init_vector
    curr_x = init_vector

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

    iterations = 0
    while norm(f - np.matmul(A, curr_x)) > epsilon:
        prev_x = curr_x
        curr_x = -np.matmul(V, prev_x) + np.matmul(w * M, f)
        iterations += 1

    return (curr_x, iterations)




def main():

    mu1 = normMat1(A) * normMat1(np.linalg.inv(A))
    print(f"Condition number in 1st norm \u03bc = {mu1}")

    mu2 = normMat2(A) * normMat2(np.linalg.inv(A))
    print(f"Condition number in 2nd norm \u03bc = {mu2}")

    mu3 = normMat3(A) * normMat3(np.linalg.inv(A))
    print(f"Condition number in 3rd norm \u03bc = {mu3}")

    print(f"Maximum eigenvalue max|\u03bb| = {findMaxEigenvalue(A)}")
    print(f"Minimum eigenvalue min|\u03bb| = {findMinEigenvalue(A)}")

    print("================================================STRAIGHT METHODS===============================================")
    if IsCholeskyCompatible(A):
        ### LL^T * x = f
        ### Ly = f, where y = L^T * x
        L = findL(A)
        y = solveForLowerTr(L, f)
        x = solveForUpperTr(L.T, y)
        

        print("My solution:")
        printVector(x)

        print(f"Residual in 1st norm = {normVec1(f - np.matmul(A, x))}")
        print(f"Residual in 2nd norm = {normVec2(f - np.matmul(A, x))}")
        print(f"Residual in 3rd norm = {normVec3(f - np.matmul(A, x))}")
        

    print("================================================ITERATIVE METHODS==============================================")
    if IsUpperRelaxationCompatible(A):
        print("Upper relaxation method\n\n")

        accuracy = 1e-6
        current_norm = normVec1
        print("My solution:")
        x, iters = UpperRelaxation(A, f, w=1.5, epsilon=accuracy, norm=current_norm)
        printVector(x)
        
        print(f"Accuracy \u03b5 = {accuracy}")
        norm_num = "₁" if current_norm is normVec1 else "₂" if current_norm is normVec2 else"₃"

        print(f"Stopping rule: ||f - Ax||{norm_num} < \u03b5")

        print(f"Iterations = {iters}")

        print(f"Residual in 1st norm = {normVec1(f - np.matmul(A, x))}")
        print(f"Residual in 2nd norm = {normVec2(f - np.matmul(A, x))}")
        print(f"Residual in 3rd norm = {normVec3(f - np.matmul(A, x))}")

    print("==================================================TRUE SOLUTION================================================")
    printVector(np.linalg.solve(A, f))
    
    eigenvals, eigenvecs = np.linalg.eig(A)
    print(f"True maximum and minimum eigenvalues respectively: \nmax|\u03bb| = {np.max(eigenvals)} \nmin|\u03bb| = {np.min(eigenvals)}")

    return 0




if __name__ == '__main__':
    main()
