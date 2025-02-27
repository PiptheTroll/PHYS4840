import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

###------------question 1-----------###

def trapezoidal(x_values, y_values, N):

    a = 0
    b = 1
    h = (b - a) / N

    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral

# Simpson's rule for array data
def simpsons(x_values, y_values, N):

    a = 0
    b = 1
    h = (b - a) / N

    integral = y_values[0] + y_values[-1]

    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling

# Romberg integration for array data
def romberg(x_values, y_values, max_order):

    R = np.zeros((max_order, max_order))
    a = 0
    b = 1
    N = 0
    h = (b - a)

    # First trapezoidal estimate
    R[0, 0] = (h / 2) * (y_values[0] + y_values[-1])

    for i in range(1, max_order):
        N = 2**i #Remember: we are recomputing the integral with different N (and therefore h)
        h = (b - a) / 2**i #Look at the github derivation for richardson extrapolation

        sum_new_points = sum(np.interp(a + k * h, x_values, y_values) for k in range(1, N, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_new_points

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]

gaia = pd.read_csv('/d/users/iank/Downloads/GAIA_G.csv', header=None, sep=',', dtype='float')
vega = pd.read_csv('/d/users/iank/Downloads/vega_SED.csv', header=[0], sep=',', dtype='float')

gwav = gaia[0]
gflux = gaia[1]
vflux = vega['FLUX']
vwav = vega['WAVELENGTH']

gwav = gwav.to_numpy()
gflux = gflux.to_numpy()
vflux = vflux.to_numpy()
vwav = vwav.to_numpy()

###------------part a-----------###

n = 20

print('romberg integration for vega is', romberg(vwav, vflux, n))
print('trapezoidal integration for vega is', trapezoidal(vwav, vflux, n))
print('simpsons integration for vega is', simpsons(vwav, vflux, n))
print('romberg integration for gaia is', romberg(gwav, gflux, n))
print('trapezoidal integration for gaia is', trapezoidal(gwav, gflux, n))
print('simpsons integration for gaia is', simpsons(gwav, gflux, n))

###------------part b-----------###

## the vega data file has its flux data written in scientific notation with capitolized Es.
## in order for the data to be read as a float value I had to edit the csv file to swap all the Es with es.
## I also had to make sure I read the file with header notation.

###------------question 2-----------###

## For the numerical integration problem we encountered in the in class lab, I would use the trapezoidal integration method. 
## The trapezoidal method is consistently one of the more accurate methods, as well as one of the shortest intgration time. 
## The evalutation time for each of the trapezoidal method follows the same pattern of increasing integration time related to the n value of the method.

###------------question 3-----------###
###------------part a-----------###

n = 3

def s(n):
	sum = 0
	x = np.arange(1, n + 1)
	for i in x:
		y = i**2
		sum += y
	return sum

print(s(n))

###------------part b-----------###

def xbar(n):
    sum = 0
    S = (np.arange(1, n + 1)**3)
    for i in S:
        sum += i

    return sum / len(S)

print(xbar(n))

###------------part c-----------###

def nfact(n):
	if n >= 1:
		sum = n * (nfact(n - 1))
	else:
		sum = 1
	return sum

print(nfact(n))

###------------question 4-----------###
###------------part a-----------###

def f(x):
	return np.sin(x)**2 / x**2

def trapezoidal_rule(f, a, b, n):
#thanks to the README.md doc on github
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

#def errcheck():
    #

#def newtrap(y):
    #x2 = x + h
    #xm = (x1 + x2) /2
    #for i in x:
        #if 
    #return 

x = np.arange(0, 10+1)

###------------part b-----------###

## this is most likely due to the fact that python struggles with maintaining the true value of numbers through math manipulation.
## If we were to simply calculate the x inputs from their function outputs, the values might not be accurate.

###------------part c-----------###

#fig, ax = plt.subplots()

#ax.plot(x, f(x), color='k')
#ax.fill_between(x, newtrap(f(x), alpha=.2))