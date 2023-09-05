import numpy as np
import random
import time
np.seterr(divide='ignore', invalid='ignore')


def defaultProblem():
    c = np.array([2, 3, 3, 1, -2])
    matrixA = np.array([[1, 3, 0, 4, 1], [1, 2, 0, -3, 1], [1, 4, -3, 0, 0]])
    b = np.array([2, 2, -1])
    print("---------------------------------------------------------------------------------------------")
    print("Show the default problem:")
    print("c", c)
    print("A", matrixA)
    print("b", b)
    print("---------------------------------------------------------------------------------------------")
    return c, matrixA, b


def fileinput(filename=r'example.txt'):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            numline = list(map(int, line.split(',')))
            data.append(numline)
    c = np.array(data[0])
    b = np.array(data[1])
    A = np.array(data[2:])
    print("---------------------------------------------------------------------------------------------")
    print("Show the file problem:")
    print("c", c)
    print("A", A)
    print("b", b)
    print("---------------------------------------------------------------------------------------------")
    return c, A, b

def mode_choice():
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
    elif testnum == 2:
        print("Please input the filename:")
        filename = input()
        c, matrixA, b = fileinput(filename)
    elif testnum == 3:
        c, matrixA, b = fileinput()
    else:
        c, matrixA, b = defaultProblem()
    return c, matrixA, b


def simplex_phase(c, A, b):
    A, b, c_B, basis_index, FFlag = simplex_phase1(c, A, b)
    if FFlag == 0:
        opt_x, opt_v = simplex_phase2(c, A, b, c_B, basis_index)
        return opt_x, opt_v
    else:
        print('This problem is unsolvable!')
        return None, None


def simplex_phase1(c, A, b):
    m, n = A.shape

    sign = 2 * (0.5 - np.signbit(b))
    A = A * sign.reshape(m, 1)
    b = np.abs(b)
    # print(c,A,b)

    basis = np.identity(m)
    c_B = np.ones(m)
    nc = np.zeros(n)

    nA = np.hstack((A, basis))
    nc = np.append(nc, c_B)
    basis_index = np.arange(n, m + n)
    # print(basis_index)
    # print(nA)
    # print(nc)
    i = 0
    while 1:
        r_cost = reduced_cost(nc, c_B, nA)
        if np.min(r_cost) >= 0:
            break
        # print(r_cost)

        # enters_index = np.argmin(r_cost)
        # print(r_cost)
        l_enter_index = np.argwhere(r_cost < 0).ravel()
        # print('l_enter_index',l_enter_index)
        enters_index = random.choice(l_enter_index)
        # print('enters_index',enters_index)
        enters = nA[:, enters_index]
        theta = b / enters
        theta_p = np.where(theta > 0, theta, np.inf)
        exits_order = np.argmin(theta_p)  # actually it is the order of row
        nA, b, c_B, basis_index = tableau(nA,b,nc,c_B,basis_index,enters_index,exits_order)
        # print(basis_index)
        # print(nA,b)
    # print(basis_index)

    # Feasibility judgment
    opt_x = np.zeros(m+n)
    opt_x[basis_index] = b
    opt_v = np.sum(opt_x * nc)
    FFlag = 0
    if opt_v > 0:
        FFlag = 1
        return A, b, c_B, basis_index, FFlag

    # force exits
    if np.any(basis_index >= n):
        l_exits_order = np.where(basis_index >= n)
        # print(basis_index,l_exits_order)

        for exits_order in l_exits_order:
            for enters_index in range(0, n):
                if enters_index not in basis_index and nA[exits_order, enters_index] != 0:
                    nA, b, c_B, basis_index = tableau(nA, b, nc, c_B, basis_index, enters_index, exits_order)

    # print(nA,b)
    # print(basis_index)
    A = nA[:, 0:n]
    print(basis_index)
    c_B = c[basis_index]
    return A, b, c_B, basis_index, FFlag


def simplex_phase2(c, A, b, c_B, basis_index):
    m, n = A.shape
    while 1:
        r_cost = reduced_cost(c, c_B, A)
        if np.min(r_cost) >= 0:
            break
        # print(r_cost)

        # enters_index = np.argmin(r_cost)
        # print(r_cost)
        l_enter_index = np.argwhere(r_cost < 0).ravel()
        # print('l_enter_index', l_enter_index)
        enters_index = random.choice(l_enter_index)
        # print('enters_index', enters_index)
        enters = A[:, enters_index]
        theta = b / enters
        theta_p = np.where(theta > 0, theta, np.inf)
        exits_order = np.argmin(theta_p)  # actually it is the order of row
        A, b, c_B, basis_index = tableau(A, b, c, c_B, basis_index, enters_index, exits_order)
    # print(basis_index)
    # print(A,b)

    opt_x = np.zeros(n)
    opt_x[basis_index] = b
    opt_v = np.sum(opt_x * c)
    return opt_x, opt_v

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
    print("""
    Choose mode:
    0: Use default problem, and disable input
    1: Execute normal mode
    2: Input file name, use file data
    3: Use example file, and disable input

    """)
    T1 = time.time()
    c, matrixA, b = mode_choice()
    # process
    T2 = time.time()
    opt_x, opt_v = simplex_phase(c,matrixA,b)
    T3 = time.time()
    print("opt_x:", opt_x)
    print("opt_v:", opt_v)
    print('Total running time', (T3-T1) * 1000, 'ms')
    print('Calculation time of simplex method:', (T3-T2) * 1000, 'ms')


if __name__ == "__main__":
    main()