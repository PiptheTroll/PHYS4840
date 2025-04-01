import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb

##which gfortran returns GNU Fortran (GCC) 11.4.1 20231218 (Red Hat 11.4.1-3)

###-------Q1---------###

##results in lab_day14.py on github

###--------Q2--------###

x0 = 0
t0 = 0
t_end = 10

t_values10, x_values10 = lb.RK4(lb.f4, x0, t0, t_end, (t_end - t0) / 10)
t_values20, x_values20 = lb.RK4(lb.f4, x0, t0, t_end, (t_end - t0) / 20)
t_values50, x_values50 = lb.RK4(lb.f4, x0, t0, t_end, (t_end - t0) / 50)
t_values100, x_values100 = lb.RK4(lb.f4, x0, t0, t_end, (t_end - t0) / 100)

plt.figure(figsize=(8, 5))
plt.plot(t_values10, x_values10, label="N = 10", color="b")
plt.plot(t_values20, x_values20, label="N = 20", color="g")
plt.plot(t_values50, x_values50, label="N = 50", color="orange")
plt.plot(t_values100, x_values100, label="N = 100", color="k")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Runge-Kutta Method 4th Order Solution for dx/dt = x^3 - sint(t)")
plt.grid(True)
plt.legend()
plt.show()

###--------Q3-------####

t_values, x_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk2_results.dat',unpack=True,skiprows=1,usecols=[0, 1])

plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="RK2 Approximation", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("RK2 Method Solution from fortran n is int")
plt.grid(True)
plt.legend()
plt.show()

t_values, x_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk2_results2.dat',unpack=True,skiprows=1,usecols=[0, 1])

plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="RK2 Approximation", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("RK2 Method Solution from fortran n is 10000")
plt.grid(True)
plt.legend()
plt.show()

###-------Q4--------###

t_values, x_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk4_results.dat',unpack=True,skiprows=1,usecols=[0, 1])

plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="RK4 Approximation", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("RK4 Method Solution from fortran n is int")
plt.grid(True)
plt.legend()
plt.show()

