import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb
import time
import os

###--------Q1--------###

RC1 = 0.01
RC2 = 0.1
RC3 = 1.0

def voltagein(x, t):
	return np.where((2 * t) % 2 == 0 , 1, -1)

def voltageout1(x, t):
	return (voltagein(x, t) - x) / RC1

def voltageout2(x, t):
	return (voltagein(x, t) - x) / RC2

def voltageout3(x, t):
	return (voltagein(x, t) - x) / RC3

tvals1, xvals1 = lb.RK4(voltageout1, 0, 0, 10, .01)
tvals2, xvals2 = lb.RK4(voltageout2, 0, 0, 10, .01)
tvals3, xvals3 = lb.RK4(voltageout3, 0, 0, 10, .01)

fig, ax = plt.subplots()

ax.plot(tvals1, xvals1, label='RC = 0.01')
ax.plot(tvals2, xvals2, label='RC = 0.1')
ax.plot(tvals3, xvals3, label='RC = 1')

ax.set_xlabel('time in seconds')
ax.set_ylabel('voltage out in volts')

ax.legend()
ax.set_title('Voltage out from Runge-Kutta for the Low-Pass filter')

plt.show()

## We expect to see voltage jumps as the voltage changes due to the charging of the capacitor, larger RC time constants result in longer charging periods which we see in the Vout graph. The pulse wave didn't make the most sense to me, but if this is whatt he pulse wave is meant to do, We expect to see this form.

###------------------###
x0 = 1
t0 = 0
t_end = 10000

time0 = time.perf_counter()

t2_vals, x2_vals = lb.RK(lb.f4, x0, t0, t_end, (t_end - t0) / 100)

time1 = time.perf_counter()

os.system('gfortran RK2_10000.f90 -o RK2.exe')
os.system('./RK2.exe')

t2_10000_values, x2_10000_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk2_10000_results.dat',unpack=True,skiprows=1,usecols=[0, 1])

time2 = time.perf_counter()

t4_vals, x4_vals = lb.RK4(lb.f4, x0, t0, t_end, (t_end - t0) / 100)

time3 = time.perf_counter()

os.system('gfortran RK4_10000.f90 -o RK4.exe')
os.system('./RK4.exe')

t4_10000_values, x4_10000_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk4_10000_results.dat',unpack=True,skiprows=1,usecols=[0, 1])

time4 = time.perf_counter()

os.system('gfortran RK2.f90 -o RK_2.exe')
os.system('./RK_2.exe')
t2_values, x2_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk2_results.dat',unpack=True,skiprows=1,usecols=[0, 1])

time5 = time.perf_counter()

os.system('gfortran RK4.f90 -o RK_4.exe')
os.system('./RK_4.exe')
t4_values, x4_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk4_results3.dat',unpack=True,skiprows=1,usecols=[0, 1])

time6 = time.perf_counter()

t4_10_vals, x4_10_vals = lb.RK4(lb.f4, x0, t0, 10, (10 - t0) / 100)

time7 = time.perf_counter()

t2_values, x2_values = np.loadtxt('/d/users/iank/PHYS4840_labs/rk2_results3.dat',unpack=True,skiprows=1,usecols=[0, 1])
t2_10_vals, x2_10_vals = lb.RK(lb.f4, x0, t0, 10, (10 - t0) / 100)

###--------Q2--------###



###--------Q3--------###

fig, ax = plt.subplots()

ax.plot(t4_10_vals, x4_10_vals, label='python rk4')
ax.plot(t2_10_vals, x2_10_vals, label='python rk2')
ax.plot(t4_values, x4_values, label='fortran rk4')
ax.plot(t2_values, x2_values, label='fortran rk2')

ax.legend()

plt.show()

## a few of the plots are overlapping, at the number of terms I use, RK4 and RK2 are roughly the same.

###--------Q4--------###

print('The time it took for fortran to run Runge-Kutta 4th order over the interval 0-10000 is', time4 - time3)
print('The time it took for python to run Runge-Kutta 4th order over the interval 0-10000 is', time3 - time2)

print('The time it took for fortran to run Runge-Kutta 4th order over the interval 0-10 is', time6 - time5)
print('The time it took for python to run Runge-Kutta 4th order over the interval 0-10 is', time7 - time6)

###--------Q5--------###

##chmod +x myfile.py changes the permissions of th efile to allow it to be executable. This is not necessary for fortran code because when we run "gfortran myfile.f90 -o myfile.exe", we are createing an a;ready executable file "myfile.exe".
