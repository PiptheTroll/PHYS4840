#!/usr/bin/python3.9.12

import numpy as np
import matplotlib.pyplot as plt


#x = np.linspace(1, 100, 500)
#y = x**2.0

## linear plot
#plt.plot(x,y, linestyle='-', color='blue', linewidth=5)
#plt.xlabel('My x-axis')
#plt.ylabel('My y-axis')
#plt.grid(True) ## turn on/off as needed
#plt.show()
#plt.close()

## log plot
#plt.plot(x,y, linestyle='-', color='red', linewidth=5)
#plt.xlabel('My x-axis')
#plt.ylabel('My y-axis')
#plt.xscale('log')  # Set x-axis to log scale
#plt.yscale('log')  # Set y-axis to log scale
#plt.grid(True) ## turn on/off as needed
#plt.show()
#plt.close()

import sys
import math

## in your functions library, which should 
## be in a different file, define the function
#
# def y(x):
# 	y = 2.0*x**3.0
# 	return y
#
# and import your functions library

sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb

# define your x values
x = np.linspace(1, 100, 500)  # x values

y = lb.y(x)

# (1) make a linear plot of y vs x
plt.plot(x,y, linestyle='-', color='blue', linewidth=5)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.grid(True) ## turn on/off as needed
plt.show()
plt.close()
# (2) make a log-log plot of y vs x
plt.plot(x,y, linestyle='-', color='red', linewidth=5)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale
plt.grid(True) ## turn on/off as needed
plt.show()
plt.close()
# (3) make a plot of log(x) vs log(y)
x2 = np.log10(x)
y2 = np.log10(y)
plt.plot(x2,y2, linestyle='-', color='green', linewidth=5)
plt.xlabel('My x-axis')
plt.ylabel('My y-axis')
plt.grid(True) ## turn on/off as needed
plt.show()
plt.close()