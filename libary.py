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