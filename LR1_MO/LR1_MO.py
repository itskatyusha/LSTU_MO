import numpy as np
import math

#1
# A = np.array([[4,6,10,2,6],[1,2,5,4,5],[10,7,10,2,4],[3,6,2,10,7],[5,1,7,10,8]])
# B = np.array([[210],[119],[241],[187],[201]])

#2
# A = np.array([[6,5,1,8],[2,8,5,9],[4,7,5,9],[3,1,1,7]])
# B = np.array([[134],[177],[179],[89]])

#3
# A = np.array([[3,1,5,3,4,8],[7,7,5,8,3,6],[10,7,10,10,7,5],[3,5,6,10,10,3],[9,4,3,8,5,1],[4,3,6,10,9,6]])
# B = np.array([[85],[135],[184],[164],[118],[151]])

#4
# A = np.array([[4,9,8,1,5,2],[10,5,8,5,4,8],[7,7,1,1,8,1],[9,10,5,1,10,9],[9,4,7,9,10,7]])
# B = np.array([[173],[218],[167],[280],[288]])
#X = np.array([[3],[ 8],[1],[1],[9],[3]])

#5
# A = np.array([[4,6,10,2,6],[1,2,5,4,5],[10,7,10,2,4],[3,6,2,10,7],[5,1,7,10,8]])
# B = np.array([[210],[119],[241],[187],[201]])

#Ручной ввод
A = np.loadtxt('matrix_A', dtype="int")
B = np.loadtxt('matrix_B', ndmin=2, dtype="int")

print('Исходная матрица A: ', '\n')
print(A,'\n')
print('Исходная матрица B: ', '\n')
print(B,'\n')

def Grevil(A):
    num_rows, num_cols = A.shape
    print('Шаг ', 1, ':', '\n')
    
    if ((~A.any(axis=0))[0] == True):
        Ak_plus = np.array([(np.zeros(num_cols))])
    else:
        Ak_plus = np.array([np.transpose(A[:, 0])/np.sum(A[:, 0]**2)])
        
    print('Ak+\n', Ak_plus, '\n')

    for i in range(1, num_cols):
        A_k = A[:, 0:i]
        print('Шаг ', i+1, '\n')
        C_k = np.array((np.dot((np.eye(num_rows) - np.dot(A_k, Ak_plus)), A[:, i:i+1])))
        
        if (np.isclose(np.sum(C_k**2), 0, rtol=1e-05, atol=1e-08, equal_nan=False) == False):
            f_k = (np.transpose(C_k)/( np.sum(C_k**2)))
        else:
            d_k = 1 + np.sum((np.dot(Ak_plus, A[:,i:i+1]))**2)
            f_k = np.dot(np.dot(np.transpose(A[:, i:i+1]), np.transpose(Ak_plus)), Ak_plus)/d_k
            
        Ak_plus = np.concatenate((np.dot(Ak_plus, np.eye(num_rows) - np.dot(A[:, i:i+1], f_k)), f_k))
        print('f_k\n', f_k, '\n')
        print('Ak+\n', Ak_plus, '\n')
    return Ak_plus

A_plus = Grevil(A)

X = np.dot(A_plus, B)
print('Решение системы\n', X, '\n')

B_0 = np.dot(A, X)

nev = np.sum((B-B_0)**2)
print('Норма невязки = ', nev)