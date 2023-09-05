import two_phase
import numpy as np
import time
import random

def simplex_dual(c, A, b):
    A, b, c_B, basis_index, FFlag = simplex_phase1(c, A, b)
    if FFlag == 0:
        opt_x, opt_v = dualmethod(c, A, b,basis_index)
        return opt_x, opt_v
    else:
        print('This problem is unsolvable!')
        return None, None


def simplex_phase1(c, A, b):
    m, n = A.shape
    A = A * (2*( 0.5-np.signbit(c)))
    print(A)
    c = np.abs(c)
    basis = np.identity(m)
    c_B = np.ones(m)
    nc = np.zeros(n)

    nA = np.hstack((A, basis))
    nc = np.append(nc, c_B)
    basis_index = np.arange(n, m + n)
    # print(nA.shape)
    # print(basis_index)
    # print(nA)
    # print(nc)
    i = 0
    while 1:
        i += 1
        if i/ 9 == 0:
            print('iter:',i* 10)
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
        # print(len(nA))
        theta = b / enters
        theta_p = np.where(theta > 0, theta, np.inf)
        # print(b.shape,enters.shape,theta.shape,theta_p.shape)
        exits_order = np.argmin(theta_p)  # actually it is the order of row
        # print(enters_index,exits_order)
        nA, b, c_B, basis_index = tableau(nA, b, nc, c_B, basis_index, enters_index, exits_order)
        # print(basis_index)
        # print(nA,b)
    # print(basis_index)

    # Feasibility judgment
    opt_x = np.zeros(m + n)
    opt_x[basis_index] = b
    opt_v = np.sum(opt_x * nc)
    FFlag = 0
    if opt_v > 0:
        FFlag = 1
        return A, b, c_B, basis_index, FFlag
    print(basis_index)
    # force exits
    if np.any(basis_index >= n):
        l_exits_order = np.where(basis_index >= n)[0]
        print(l_exits_order)

        for exits_order in l_exits_order:
            # print(exits_order)
            for enters_index in range(0, n):
                print(nA[exits_order, enters_index])
                if enters_index not in basis_index and nA[exits_order, enters_index] != 0:
                    nA, b, c_B, basis_index = tableau(nA, b, nc, c_B, basis_index, enters_index, exits_order)
                    print("force_swap:",basis_index)

    # print(nA,b)
    # print(basis_index)
    A = nA[:, 0:n]
    # print(basis_index)
    c_B = c[basis_index]
    return A, b, c_B, basis_index, FFlag


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


def dualmethod(c, A, b, basis_index):
    while 1:
        c_B = c[basis_index]
        r_cost = two_phase.reduced_cost(c,c_B,A)
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
        A, b, c_B, basis_index = two_phase.tableau(A, b, c, c_B, basis_index, enters_index, exits_order)


def main():
    # input
    # random.seed(1234)
    T1 = time.time()
    c, matrixA, b = two_phase.mode_choice()
    # process
    T2 = time.time()
    opt_x, opt_v = simplex_dual(c,matrixA,b)
    T3 = time.time()
    print("opt_x:", opt_x)
    print("opt_v:", opt_v)
    print('Total running time', (T3-T1) * 1000, 'ms')
    print('Calculation time of simplex method:', (T3-T2) * 1000, 'ms')

if __name__ == '__main__':
    main()