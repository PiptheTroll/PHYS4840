import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def gaussian(x, A, B, C, D, E):
#A,B,C,D,E are the changeable paramaters, use an array for the x values.
	return A + (B * x) + (C * (np.exp(-(x - D)**2 / (2 * E**2))))

def format_axes(ax):
##creates a more visually appealing axis design
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)  # Larger major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=1)    # Minor ticks
    ax.minorticks_on()  # Enable minor ticks

def distance_mod(p):
##calculates the distance modulus (m - M) with p in parsecs 
	return 5 * np.log10(p/10)

def f1(x):
## 1 + tanh(2x)/2 function used starting in lab_day10
    return 1 + (np.tanh(2 * x) / 2)

def f2(x):
## derivative of the f1 functions as a function
    return 1 / np.cosh(2 * x)**2