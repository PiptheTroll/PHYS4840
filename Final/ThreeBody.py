#! user/bin/
###----------imports----------###

import numpy as np
import astropy
import time
import ThreeBody_lib as lib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

###---------------------------###
animate = True
###--------Updateables--------###

dt = (.1) #size of time steps in seconds
T = (100.) #length of simulation in seconds
safe_r = .1 #forces particles away from other particles at this radius, in m

m1 = 2500000. #mass of first obect, in kg
m2 = 3500000000. #mass of second object, in kg
m3 = 10000000.#mass of thd objirect, in kg

#Initial positions and velocities for each object m and m/s
p_initial_1 = np.array([2., 1., -1.])
p_initial_2 = np.array([0., -1., 0.])
p_initial_3 = np.array([0., 0., 3.])
v_initial_1 = np.array([-.5, 0., 0.])
v_initial_2 = np.array([-.1, 0., 0.])
v_initial_3 = np.array([.1, 0., 0.])

###--------------------------###

###--------Main--------------###

def main():
# Main chunk to simulate and animate the model

# Quick definition of the length of the simulation in number of steps
	length = int(T / (dt))

# Creates the array to store Position variables 
	positions_1 = np.zeros((length + 1, 3)) 
	positions_2 = np.zeros((length + 1, 3))
	positions_3 = np.zeros((length + 1, 3))

# Creates the array to store velocity variables
	velocities_1 = np.zeros((length + 1, 3)) 
	velocities_2 = np.zeros((length + 1, 3)) 
	velocities_3 = np.zeros((length + 1, 3)) 

# Sets the initial positions of the bodies in m from initial conditions
	positions_1[0] = p_initial_1
	positions_2[0] = p_initial_2
	positions_3[0] = p_initial_3

# Sets the initial velocities in m/s from initial conditions
	velocities_1[0] = v_initial_1 
	velocities_2[0] = v_initial_2 
	velocities_3[0] = v_initial_3 

	print('Starting Simulation')

# Main loop for simulating the body locations
	for i in range (1, length+1):

		time1 = time.perf_counter()

# Calculates the acceleration at each point
		a12, a21 = lib.find_acc(positions_1[i - 1], positions_2[i - 1], m1, m2)
		a23, a32 = lib.find_acc(positions_2[i - 1], positions_3[i - 1], m2, m3)
		a13, a31 = lib.find_acc(positions_1[i - 1], positions_3[i - 1], m1, m3)

		a1_tot = (a12 + a13)
		a2_tot = (a23 + a21)
		a3_tot = (a32 + a31)

# uses the acceleration to find the new positions and velocities on time step, dt, forward, and updates the data array
		positions_1[i], velocities_1[i] = lib.find_new_p(positions_1[i - 1], velocities_1[i - 1], a1_tot, dt)
		positions_2[i], velocities_2[i] = lib.find_new_p(positions_2[i - 1], velocities_2[i - 1], a2_tot, dt)
		positions_3[i], velocities_3[i] = lib.find_new_p(positions_3[i - 1], velocities_3[i - 1], a3_tot, dt)
		
		if lib.find_norm(positions_1[i] - positions_2[i]) < safe_r:
			positions_1[i], positions_2 = lib.force_field(positions_1[i], positions_2[i], m1, m2, safe_r)
		if lib.find_norm(positions_2[i] - positions_3[i]) < safe_r:
			positions_2[i], positions_3 = lib.force_field(positions_2[i], positions_3[i], m2, m3, safe_r)
		if lib.find_norm(positions_1[i] - positions_3[i]) < safe_r:
			positions_1[i], positions_3[i] = lib.force_field(positions_1[i], positions_3[i], m1, m3, safe_r)
		
		time2 = time.perf_counter()

		if i==1:
			print("expected time remaining:", (time2 - time1) * (length - 1), 'seconds')

	np.savetxt("ThreeBody1.dat", positions_1, delimiter=',')
	np.savetxt("ThreeBody2.dat", positions_2, delimiter=',')
	np.savetxt("ThreeBody3.dat", positions_3, delimiter=',')

	#np.savetxt("ThreeBodyv1.dat", velocities_1, delimiter=',')
	#np.savetxt("ThreeBodyv2.dat", velocities_2, delimiter=',')
	#np.savetxt("ThreeBodyv3.dat", velocities_3, delimiter=',')

	print('Simulation Finished')
	if animate is True:
		print('Animating Simulation')

		#lib.animate_3body(positions_1, positions_2, positions_3, dt, T)
		lib.animate_3body('ThreeBody1.dat', 'ThreeBody2.dat', 'ThreeBody3.dat', dt, T)

		print('Simulation Animated')
	print('Have a Great Day')



###--------------------------###

if __name__ == "__main__":
	main()
