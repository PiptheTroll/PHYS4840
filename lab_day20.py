import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import empty,zeros,max
from pylab import plot, xlabel, ylabel, show

# Constants

L=0.01 # Thickness of steel in meters 
D = 4.25e-6 # Thermal diffusivity
N = 100 # Number of divisions in grid 
a = L / N # Grid spacing
h = 1e-4 # Time-step
epsilon = h / 1000

Tlo = 0.0 # Low temperature in Celsius 
Tmid = 20.0 # Intermediate temperature in Celsius 
Thi = 50.0 # Hi temperature in Celsius

t1 = 0.01 
t2 = 0.1 
t3 = 0.4 
t4 = 1.0 
t5 = 10.0

tend = t5 + epsilon

#Create arrays

T = empty (N+1,float) 

T[0] = Thi

T[N] = Tlo


T[1:N] = Tmid

Tp = empty(N+1,float)

Tp[0] = Thi 
Tp[N] = Tlo # Main loop 
t=0.0

c = h * D / (a * a) 

while t < tend:
# Calculate the new values of T 
	Tp[1:N] = T[1:N] + c * (T[0:N-1] + T[2:N+1] - 2 * T[1:N]) 
	T,Tp = Tp,T 
	t += h

# Make plots at the given times 
	if abs(t - t1) < epsilon:
		plot (T) 
	if abs(t - t2) < epsilon: 
		plot (T) 
	if abs(t - t3) < epsilon: 
		plot (T) 
	if abs(t - t4) < epsilon: 
		plot (T) 
	if abs(t - t5) < epsilon: 
		plot (T) 
	
xlabel ("x") 
ylabel("T")

show()
