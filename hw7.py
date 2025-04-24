import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb
import time
import numpy as np
import matplotlib.pyplot as plt

from pylab import plot, xlabel, ylabel, show

###--------------Question 0--------------###

##Answers to lab day 18 and 19

##Q1
## a ## There are now 3 dimensions as opposed to two, so each vector needs an additional scalar value for the third dimension.
## b ## It takes way too long ot process if the frame size is too large.
## c ## It adds additional steps when you increase the accuracy. Each step takes it closer to the accuracy you want, so the more accurate you want the longer it takes.
## d ## In the 2D case, the boundary conditions are maintained at the start of the formulation. In the 3D case, the boundary conditions are forced after the formulation occurs.

###--------------Question 1--------------###

D = .1 #m^2day^-1
B = 273.15 + 12
A = 273.15 + 10
tau = 365 #days

N = 3650

h = 1
a = 1

c = h * D / (a * a)

#T = A + B * np.sin(2 * np.pi * t / tau)
def dT_dt_D(A, B, tau, D, t):
	return (B * 2 * np.pi / tau) * np.cos(2 * np.pi * t / tau) / D

t = np.arange(0, 365 * 10, 1)
T = np.empty((3650), float)
Tp = np.empty((3650), float)

T_nought = dT_dt_D(A, B, tau, D, t)
t_0 = 0.0

tend = N + 1

while t_0 < tend:
# Calculate the new values of T 
	for i in range(1,N - 1): 
		Tp[i] = T_nought[i] + c*(T_nought[i+1] + T_nought[i - 1] - 2 * T_nought[i]) 
	T,Tp = Tp,T 
	t_0 += h

	if t_0 >= (tend - 365) and t_0 <= (tend - 3 * 365 / 4):
		plot(T)
	if t_0 > (tend - 3 * 365 / 4) and t_0 <= (tend - 2 * 365 / 4):
		plot(T)
	if t_0 > (tend - 2 * 365 / 4) and t_0 <= (tend - 365 / 4):
		plot(T)
	if t_0 > (tend - 365 / 4) and t_0 < tend:
		plot(T)


xlabel('x')
ylabel('T')
show()

###--------------Question 2-------------###

## In "Modules for Experiments in Stellar Astrophysics (MESA): Planets, Oscillations, Rotation, and Massive Stars" on page page 19, the paper describes using the diffusion equation with pdes. They use it to solve for the transportof angular momentum in the medium of the stars as the star spins. The diffusion equation is primarily used for heat diffusion, but this method still works for diffusion of anything through a medium.

###--------------Question 3--------------###

#  The fortran code, stellar_RK4.f90 uses the mass coservation module to solve the equations of the stellar structure. The initial conditions are 0 mass at 0 radius. The boundary conditions are enforced by starting the loop with the initial conditions asa known, then building off of those initial conditions. 

r, m = np.loadtxt('/d/users/iank/PHYS4840_labs/profile.dat',unpack=True,skiprows=1,usecols=[0, 1])

fig, ax = plt.subplots()

ax.scatter(r, m, label='stellar mass')

ax.set_title('stellar mass as a function of stellar radius')
ax.legend()

plt.show()

###--------------Question 4--------------###

## Parabolic, boundary condiitions have no propagation time and effect all points of time the same. An example of this is the Schrodinger equation. Like all the rest of the examples, it can be solved with the method of finite differences.
## Elliptic, boundary conditions propagate through time in a steady and even way. An example of this is the Laplace equation.
## Hyperbolic, boundary conditions propagate in an uneven waveform through time. An example of this are the Maxwell equations.


