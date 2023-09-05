import numpy as np
import time
import random
import scipy.io as scio

def defaultProblem():
    c = np.array([-3, 1, 0, 0, 0])
    matrixA = np.array([[1, 0, 1/7, 0, 2/7], [0, 1, -2/7, 0, 3/7], [0, 0, -3/7, 0, 22/7]])
    b = np.array([13/7, 9/7, 31/7])
    basis_index = np.array([0,1,3])
    print("---------------------------------------------------------------------------------------------")
    print("Show the default problem:")
    print("c", c)
    print("A", matrixA)
    print("b", b)
    print("---------------------------------------------------------------------------------------------")
    return c, matrixA, b, basis_index


def fileinput(filename=r'example.txt'):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            numline = list(map(int, line.split(',')))
            data.append(numline)
    c = np.array(data[0])
    b = np.array(data[1])
    basis_index = np.array(data[2])
    A = np.array(data[3:])
    print("---------------------------------------------------------------------------------------------")
    print("Show the file problem:")
    print("c", c)
    print("A", A)
    print("b", b)
    print("basis_index", basis_index)
    print("---------------------------------------------------------------------------------------------")
    return c, A, b, basis_index


def mat_fileinput(filename):
    data = scio.loadmat(filename)
    A = data['A'].A
    b = data['b'].ravel()
    c = data['c'].ravel()
    print("---------------------------------------------------------------------------------------------")
    print("Show the file problem:")
    # print("c", c)
    # print("A", A)
    # print("b", b)
    print('c','type:',type(c),'length:',len(c))
    print('b','type:',type(b),'length:',len(b))
    print('A','type:',type(A),'shape:',A.shape)
    print("basis_index", basis_index)
    print("---------------------------------------------------------------------------------------------")
    return c, A, b

def mode_choice():
    print("""
    Choose mode:
    0: Use default problem, and disable input
    1: Execute normal mode
    2: Input file name, use file data,"xxx.txt"
    3: Use example file, and disable input
    """)
    testnum = int(input("Please input test number:"))

    if testnum == 1:
        print('Please input the c(such as "1,2,3,4",0 cannot be omitted):')
        c = np.array(list(map(int, input().split(','))))
        print('Please input the size of A:( "2,4"):')
        m, n = map(int, input().split(','))
        # print(c,m,n)
        matrixA = np.zeros((m, n))
        if n != len(c):
            print("Error!!! The number of x is different from the number of column.")
            return 1
        for i in range(m):
            print('Please input the %d row(such as "1,2,3,4",0 cannot be omitted):' % i)
            row = np.array(list(map(int, input().split(','))))
            matrixA[i, :] = row
        # print(matrixA)
        print('Please input the b(such as "1,2,3,4",0 cannot be omitted):')
        b = np.array(list(map(int, input().split(','))))
        print('Please input the basis_index(such as "1,2,3,4",0 cannot be omitted):')
        basis_index = np.array(list(map(int, input().split(','))))
    elif testnum == 2:
        print("Please input the filename:")
        filename = input()
        c, matrixA, b, basis_index = fileinput(filename)
    elif testnum == 3:
        c, matrixA, b,basis_index = fileinput()
    # elif testnum == 4:
    #     print("Please input the filename:")
    #     filename = input()
    #     c, matrixA, b = mat_fileinput(filename)
    else:
        c, matrixA, b, basis_index = defaultProblem()
    return c, matrixA, b, basis_index


# def simplex_dual(c, A, b, basis_index):
#     opt_x, opt_v = dualmethod(c, A, b,basis_index)
#     return opt_x, opt_v
#     else:
#         print('This problem is unsolvable!')
#         return None, None
#
# def simplex_phase1(c, A, b):


def dualmethod(c, A, b, basis_index):
    while 1:
        c_B = c[basis_index]
        r_cost = reduced_cost(c,c_B,A)
        if np.min(r_cost) < 0:
            print('This problem input is illegal!')
            return None, None
        if np.min(b) >= 0:
            opt_x = np.zeros(len(c))
            opt_x[basis_index] = b
            opt_v = np.sum(opt_x * c)
            return opt_x, opt_v

        l_exits_index = np.argwhere(b < 0).ravel()
        exits_index = random.choice(l_exits_index)
        exits_order = np.where(basis_index == exits_index)
        exits = A[exits_order, :]

        if np.min(exits) >= 0:
            print('the optimal dual cost is positive infinite, the primal problem is infeasible!')
            return None, None

        theta = r_cost / exits
        theta_p = np.where(theta < 0, theta, np.inf)
        enters_index = np.argmin(np.abs(theta_p))
        A, b, c_B, basis_index = tableau(A, b, c, c_B, basis_index, enters_index, exits_order)


def tableau(A,b,c,c_B,basis_index,enters_index, exits_order):
    m,n = A.shape
    enters = A[:, enters_index]
    c_B[exits_order] = c[enters_index]
    # print("c_B:",c_B,"b:",b)
    tmp_b = b[exits_order] / enters[exits_order]
    tmp_r = A[exits_order, :] / enters[exits_order]
    # print(tmp_r,tmp_b)
    p1 = enters / enters[exits_order]
    b = b - b[exits_order] * p1
    A = A - A[exits_order] * p1.reshape(m, 1)
    # print(nA,b)
    b[exits_order] = tmp_b
    A[exits_order, :] = tmp_r
    basis_index[exits_order] = enters_index
    return A, b, c_B, basis_index

def reduced_cost(c, c_B, M):
    m, n = M.shape
    r_cost = np.zeros(n)
    c_B_column = c_B.reshape(m, 1)
    # print(np.sum(c_B_column * M,axis=0))
    r_cost = c - np.sum(c_B_column * M, axis=0)
    return r_cost


def main():
    # input
    random.seed(1234)
    T1 = time.time()
    c, matrixA, b, basis_index = mode_choice()
    # process
    T2 = time.time()
    opt_x, opt_v = dualmethod(c,matrixA,b, basis_index)
    T3 = time.time()
    print("opt_x:", opt_x)
    print("opt_v:", opt_v)
    print('Total running time', (T3-T1) * 1000, 'ms')
    print('Calculation time of simplex method:', (T3-T2) * 1000, 'ms')

if __name__ == '__main__':
    main()