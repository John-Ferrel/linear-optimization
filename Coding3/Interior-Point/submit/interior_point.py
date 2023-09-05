import numpy as np
import time


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


def make_A_dash(A,S,X,nv,ne):
	z1 = np.zeros((nv,nv))
	z2 = np.zeros((nv,ne))
	row1 = np.concatenate((A,z1),axis=1)
	row1 = np.concatenate((row1,z2),axis=1)
	# print(row1.shape)

	z3 = np.zeros((ne,ne))
	I = np.eye(ne)
	row2 = np.concatenate((z3,A.T,),axis=1)
	row2 = np.concatenate((row2,I),axis=1)
	# print(row2.shape)

	row3 = np.concatenate((S,z2.T),axis=1)
	row3 = np.concatenate((row3,X),axis=1)
	# print(row3.shape)

	M = np.concatenate((row1,row2),axis=0)
	M = np.concatenate((M,row3),axis=0)

	return M

def make_b_dash(A,b,c,x,y,s,mu,nv,ne):
	e = np.ones(ne)
	X = np.diag(x)
	S = np.diag(s)
	XSe = np.matmul(np.matmul(X,S),e)

	sigma = 0.2

	row1 = b - np.matmul(A,x)
	row2 = c - np.matmul(A.T,y) - s
	row3 = sigma*mu*e - XSe

	b = np.concatenate((row1,row2),axis=None)
	b = np.concatenate((b,row3),axis=None)

	return b

def alpha_set(x,s,del_x,nv,ne,alpha):
	x_dash = del_x[:ne]
	y_dash = del_x[ne:nv+ne]
	s_dash = del_x[nv+ne:]

	alpha_x = []
	alpha_s = []

	# print(s,s_dash)
	# print(x,x_dash)

	for i in range(x.shape[0]):
		if x_dash[i]<0:
			alpha_x.append(x[i]/-x_dash[i])
		if s_dash[i]<0:
			alpha_s.append(s[i]/-s_dash[i])

	if len(alpha_x)==0 and len(alpha_s)==0:
		return alpha
	else:
		alpha_x.append(np.inf)
		alpha_s.append(np.inf)
		alpha_x = np.array(alpha_x)
		alpha_s = np.array(alpha_s)

		alpha_max = min(np.min(alpha_x), np.min(alpha_s))

		eta = 0.999
		alpha_k = min(1,eta*alpha_max)

	return alpha_k


def interior_point(A,b,c,nv,ne):
	x = np.ones(ne)
	y = np.ones(nv)
	s = np.ones(ne)

	alpha = 0.5
	epsilon = 1e-10

	while(True):
		mu = np.dot(x,s)/ne
		X = np.diag(x)
		S = np.diag(s)
		A_dash = make_A_dash(A,S,X,nv,ne)
		b_dash = make_b_dash(A,b,c,x,y,s,mu,nv,ne)

		# print('A_dash',A_dash)
		if mu<epsilon or x.all()<0 or s.all()<0:
			return x

		del_x = np.linalg.solve(A_dash,b_dash)

		alpha = alpha_set(x,s,del_x,nv,ne,alpha)

		x = x + alpha*del_x[:ne]
		y = y + alpha*del_x[ne:ne+nv]
		s = s + alpha*del_x[ne+nv:]

		# print(alpha)

def main():
	# input
	print("""
	Choose mode:
	0: Use default problem, and disable input
	1: Execute normal mode, input in the window
	2: Input file name, use file data
	3: Use example file, and disable input

	""")
	T1 = time.time()
	c, A, b = mode_choice()
	# process
	T2 = time.time()
	x = np.zeros(A.shape[1])
	opt_x = interior_point(A,b,c,A.shape[0],A.shape[1])
	T3 = time.time()
	print("opt_x:", opt_x)
	print("opt_v:", np.dot(c,opt_x))
	print('Total running time', (T3-T1) * 1000, 'ms')
	print('Calculation time of simplex method:', (T3-T2) * 1000, 'ms')

if __name__ == '__main__':
	main()