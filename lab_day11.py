import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# some data
x = np.arange(0, 11)
y = np.sin(x)  

# Define fine-grained x-values for interpolation
x_domain = np.linspace(min(x), max(x), 100)

# Linear Interpolation
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_domain)

# Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)
y_cubic = cubic_spline(x_domain)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points', zorder=3)
plt.plot(x_domain, y_linear, '--', label='Linear Interpolation', linewidth=2)
plt.plot(x_domain, y_cubic, label='Cubic Spline Interpolation', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs. Cubic Spline Interpolation')
plt.grid(True)
plt.show()

from math  import tanh, cosh

import sys
import libary as lb

## compute the instantaneous derivatives
## using the central difference approximation
## over the interval -2 to 2

x_lower_bound = -2.0
x_upper_bound = 2.0

N_samples = 100

#####################
#
# Try different values of h
# What did we "prove" h should be
# for C = 10^(-16) in Python?
#
#######################

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)
def vals(xdata, h):
	central_diff_values = []
	for x in xdata:
		central_difference = ( lb.f1(x + 0.5*h) - lb.f1(x - 0.5*h) ) / h
		central_diff_values.append(central_difference)

	print(h)
## Add the analytical curve
## let's use the same xdata array we already made for our x values

	analytical_values = []
	for x in xdata:
		dfdx = lb.f2(x)
		analytical_values.append(dfdx)

	return central_diff_values, analytical_values

cd1 = vals(xdata, 1)[0]
cdbst = vals(xdata, 1e-10)[0]
cdbst0 = cdbst[0]
print(cdbst0)
cd2 = vals(xdata, 2)[0]
cdeneg16 = vals(xdata, 1e-16)[0]
aval = vals(xdata, 1)[1]

plt.plot(xdata, aval, linestyle='-', color='black')
plt.plot(xdata, cdbst, "*", color="green", markersize=8, alpha=0.5, label='h is 1e-10')
plt.plot(xdata, cd1, "*", color="orange", markersize=8, alpha=0.5, label='hi is 1')
plt.plot(xdata, cd2, "*", color="blue", markersize=8, alpha=0.5, label='h is 2')
plt.plot(xdata, cdeneg16, "*", color="red", markersize=8, alpha=0.5, label='h is 1e-16')
plt.savefig('numerical_vs_analytic_derivatives.png')
plt.legend()
plt.show()