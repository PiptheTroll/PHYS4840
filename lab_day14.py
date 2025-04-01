import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb

# Initial conditions
x0 = 0.5
t0 = 0
t_end = 10
n = 10000
dt = ((t_end - t0) / n) ## try two other step sizes

# Solve using Euler method
t_values, x_values = lb.euler_method(lb.f3, x0, t0, t_end, dt)

# Plotting the solution
plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="Euler Approximation", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Euler Method Solution for dx/dt = x² - x")
plt.grid(True)
plt.legend()
plt.show()

x0 = 0
t0 = 0
t_end = 10
n = 10000
dt = 1 ## try two other step sizes

# Solve using Euler method
t_values10, x_values10 = lb.RK(lb.f4, x0, t0, t_end, (t_end - t0) / 10)
t_values20, x_values20 = lb.RK(lb.f4, x0, t0, t_end, (t_end - t0) / 20)
t_values50, x_values50 = lb.RK(lb.f4, x0, t0, t_end, (t_end - t0) / 50)
t_values100, x_values100 = lb.RK(lb.f4, x0, t0, t_end, (t_end - t0) / 100)

plt.figure(figsize=(8, 5))
plt.plot(t_values10, x_values10, label="N = 10", color="b")
plt.plot(t_values20, x_values20, label="N = 20", color="g")
plt.plot(t_values50, x_values50, label="N = 50", color="orange")
plt.plot(t_values100, x_values100, label="N = 100", color="k")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Runge-Kutta Method Solution for dx/dt = x^3 - sint(t)")
plt.grid(True)
plt.legend()
plt.show()
