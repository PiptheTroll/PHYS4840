#!/usr/bin/python3.8
#####################################
#
# Class 14: Matrices and Linear algebra 
# Author: M Joyce
#
#####################################
import numpy as np
from numpy import array,empty

###------------Tuesday Lab class---------------###

##part 1##

A = array([[2, 1, 4, 1], 
            [3, 4, -1, -1], 
            [1, -4, 1, 5], 
            [2, -2, 1, 3]], float)

## dimension 
N = len(A)

# Initialize L as the N=4 identity matrix 
L = np.array([[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)])
# this above is just a more explicit way of doing
#L = np.identity(N)

print("L looks like this: ", L) ## should return the N=4 I


# initalize U as a copy of A
U = A.copy()


## this double loop will transform L
## into the lower-diagonal form we need
for m in range(N):
    for i in range(m+1, N):        
        
        # Compute the multiplier for the current row operation
        L[i, m] = U[i, m] / U[m, m]
        
        # Subtract the appropriate multiple of the pivot row from the current row
        U[i, :] -= L[i, m] * U[m, :]

print('The lower triangular matrix L is:\n', L)
print('The upper triangular matrix U is:\n', U)

def vector(a, b, c, d):
	return np.array([a, b, c, d])

b = vector(1, 2, 3, 4)

x4 = b[3] / U[3][3]
x3 = (b[2] - x4 * U[2][3]) / U[2][2]
x2 = ((b[1] - x3 * U[1][2] - x4 * U[1][3])) / U[1][1]
x1 = ((b[0] - x2 * U[0][1] - x3 * U[0][2] - x4 * U[0][3])) / U[0][0]

x = np.array([x1, x2, x3, x4])

print('x is', x)

##part 2##

vector = np.array([-4,3,9,7],float)

## dimension 
N = len(vector)

for m in range(N):

	## first, divide by the diagonal element
	divisor = A[m,m]

	## divide every entry in row m by the divisor
	A[m,:] /= divisor

	## the above is shorthand for this operation:
	## A[m,:] = A[m,:]/divisor

	##anything we do to the matrix we must do to the vector:
	vector[m] /= divisor

	## now subtract multipls of the top row from the lower rows
	## to zero out rows 2,3 and 4
	for i in range(m+1, N): ## note that we start from the second row: m+1

		## because the first row now has 1 in the upper-left corner,
		## the factor by which we have to multiply the first row to subtract
		## it from the second is equal to the value in the first entry
		## of the second row
		multiplication_factor = A[i,m] 

		## now we must apply this operation to the entire row 
		## AND vector, as usual 
		A[i,:]    -= multiplication_factor*A[m,:]
		vector[i] -= multiplication_factor*vector[m] 


print('the upper diagonal version of A is: \n', A)
print('the upper diagonal version of vector is: \n', vector)

x4 = vector[3]
x3 = (vector[2] - x4 * A[2][3])
x2 = ((vector[1] - x3 * A[1][2] - x4 * A[1][3]))
x1 = ((vector[0] - x2 * A[0][1] - x3 * A[0][2] - x4 * A[0][3]))

xx = np.array([x1, x2, x3, x4])

print('xx is', xx)

###--------------Thursday Lab class-------------###

A = np.array([ [2, -1, 3,],\
			   [-1, 4, 5], 
			   [3,  5, 6] ],float)

eigenvector_1 =  np.array([-0.5774,\
						   -0.5774,\
						   0.5774],float)

LHS = np.dot(A, eigenvector_1)

## Bonus: Why doesn't this line work??
#LHS = A*eigenvector_1

RHS = -2.0*eigenvector_1

print("LHS:\n",LHS, "\n\nRHS:\n",RHS)

def qr_decomposition(A):
    ## Computes the QR decomposition of matrix A using
    ## Gram-Schmidt orthogonalization.
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]  # Take column j of A
        for i in range(j):  # Subtract projections onto previous Q columns
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)  # Compute norm
        Q[:, j] = v / R[j, j]  # Normalize

    return Q, R

print(qr_decomposition(A))
