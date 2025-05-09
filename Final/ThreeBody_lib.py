import numpy as np
import astropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def find_norm(vector):
	norm = np.sqrt(np.sum(np.square(vector)))
	return norm

def find_acc(p1, p2, m1, m2):
# inputs of position from origin in a r dimensional array in au, and masses of objects in solar masses
# returns 3 dimensional arrays of acceleration in au/s
	G =  6.67408e-11 # (m^2 / kg) * (m / s^2)

	r = (p2 - p1)
	R = find_norm(r)
	rhat = r / R

	a1 = G * m2 * rhat / R**2
	a2 = G * m1 * -rhat / R**2

	return a1, a2

def find_new_p(initial_position, velocity, acceleration, dt):
# inputs of 3 dimensional position of the object in au, 3 dimensional velocity of object in m/s, and acceleration array in m/s^2
# returns 3 dimensional position array in au, and 3 dimensional velocity array of object
	new_position = ((initial_position) + (velocity * dt) + (acceleration * (dt)**2) / 2) 
	new_velocity = ((velocity ) + (acceleration * dt)) 

	return new_position, new_velocity

def animate_3body(positions_1, positions_2, positions_3, dt, T):
# Bulk of the animation code, uses input .dat files made previously for the x,y,z coordinates of each body to animate the frames at (fps * dt) x the speed
# returns nothing but creates a gif animation in the directory
	x1, y1, z1 = np.loadtxt(f'{positions_1}',unpack=True,skiprows=0,delimiter=',', usecols=[0, 1, 2])
	x2, y2, z2 = np.loadtxt(f'{positions_2}',unpack=True,skiprows=0,delimiter=',', usecols=[0, 1, 2])
	x3, y3, z3 = np.loadtxt(f'{positions_3}',unpack=True,skiprows=0,delimiter=',', usecols=[0, 1, 2])

	frames = int(T / (dt))


	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
# The animation function itself, stitches each frame
	def animate(i):
		ax.clear()
		#ax.set_ylim(-3, 3)
		#ax.set_xlim(-3, 3)
		#ax.set_zlim(-3, 3)
		ax.scatter(x1[:i], y1[:i], z1[:i], label = 'Body 1')
		ax.scatter(x2[:i], y2[:i], z2[:i], label = 'Body 2')
		ax.scatter(x3[:i], y3[:i], z3[:i], label = 'Body 3')
		ax.legend(loc = 'upper left')


	ani = animation.FuncAnimation(fig, animate, interval = frames, save_count=frames)
# fps defines the rate at which the animation is animated, for longer periods higher fps is recommended for more reasonable gif file sizes
	fps = 10
	ani.save('ThreeBody.gif', writer='pillow', fps=fps)


def force_field(position1, position2, m1, m2, safe_r):
# takes input of masses and positions of 2 bodies and returns safe coordinates for each
# Keeps the bodies from getting too close and launching too hard away
# Keeps the more massive object in place and displaces the less massive object to a safe distance, safe_r

	vector = position1 - position2
	unit_vector = vector / (find_norm(vector))
# space is the distance and direction the body has to move to escape the the other body
	space = unit_vector * (safe_r - find_norm(vector))

	if m1>m2:
		safe_p1 = position1 - space
		safe_p2 = position2
	else:
		safe_p2 = position2 + space
		safe_p1 = position1

	return safe_p1, safe_p2
