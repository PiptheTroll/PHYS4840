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

def f1(x, t):
## 1 + tanh(2x)/2 function used starting in lab_day10
    return 1 + (np.tanh(2 * x) / 2)

def f2(x, t):
## derivative of the f1 functions as a function
    return 1 / np.cosh(2 * x)**2

def f3(x, t):
    return x**2 - x

def f4(x, t):
    return -x**3 + np.sin(t)

def euler_method(f, x0, t0, t_end, dt):
#solves a FO ODE f, with initial values x0 at time t0, up until the time t_end, with step sizes (h) of size dt
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + dt * f(x_values[i - 1], t_values[i - 1])

    return t_values, x_values

def RK(f, x0, t0, t_end, dt):
#Runge-Kutta method for ODE DEQ solving
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + dt * f(x_values[i - 1] + .5 * (dt * f(x_values[i - 1], t_values[i - 1])), t_values[i - 1] + .5 * dt)

    return t_values, x_values

def RK4(f, x0, t0, t_end, dt):
#Runge-Kutta method for ODE DEQ solving to the fourth order
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value
    def k1(x, t, f, dt):
        return dt * f(x, t)
    def k2(x, t, f, dt):
        return dt * f(x + k1(x, t, f, dt) / 2, t + dt / 2)
    def k3(x, t, f, dt):
        return dt * f(x + k2(x, t, f, dt) / 2, t + dt / 2)
    def k4(x, t, f, dt):
        return dt * f(x + k3(x, t, f, dt), t + dt)

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + (k1(x_values[i - 1], t_values[i - 1], f, dt) + k2(x_values[i - 1], t_values[i - 1], f, dt) + k3(x_values[i - 1], t_values[i - 1], f, dt) + k4(x_values[i - 1], t_values[i - 1], f, dt)) / 6

    return t_values, x_values
