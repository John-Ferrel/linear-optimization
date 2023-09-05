import numpy as np

# A = np.array([[1,1,1],[1,1,1]])
# c_b = np.array([1,2])
# c_B_col = c_b.reshape(2,1)
# B = A * c_B_col
# C = np.sum(B,axis=0)
# print(B)
# print(C)

a = np.array([-1,1,0])
b =2*( 0.5-np.signbit(a))
c = np.sign(a)
print(b,c)
d = np.array([[1,1,1],[2,2,2]])
print(d * b)

# A = np.array([[2,2],[1,1]])
# s = np.array([0,1])
# print(A * s)
# s = s.reshape(2,1)
# print(A * s)

# if -0.0 == 0:
#     print(2)
# a = np.array([1,2,3])
# b = 1 /a
# print(b)

# a = np.array([1,1])
# b = np.array([0,1])
# c = a/b
# print(c)
# d =np.nan_to_zero(c)
# print(c,d)

#
# def reduced_cost(c, c_B, M):
#     m, n = M.shape
#     r_cost = np.zeros(n)
#     c_B_column = c_B.reshape(m, 1)
#     # print(np.sum(c_B_column * M,axis=0))
#     r_cost = c - np.sum(c_B_column * M, axis=0)
#     return r_cost
#
# A = np.array([[0,-2,-4,-1,1],[1,-1,6,-1,0]])
# c_B =
# def astring(i):
#     if i == 0:
#         return '1234','1234'
#     else:
#         return 1
#
# if astring(i = 0) == 1:
#     print(1)
a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
print(a/b)