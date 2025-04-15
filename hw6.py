import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

###------------------------------------------problem 0---------------------------###
"""
g = 9.8126  # Gravity (m/s^2)
l = 0.40   # Length of pendulum arms (m)
m = 1   # Mass of pendulums (kg)

random.seed()

theta1_1 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
theta2_1 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
omega1_1 = random.uniform(-.5, .5)  # Small random initial velocity
omega2_1 = random.uniform(-.5, .5)  # Small random initial velocity

theta1_2 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
theta2_2 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
omega1_2 = random.uniform(-.5, .5)  # Small random initial velocity
omega2_2 = random.uniform(-.5, .5)  # Small random initial velocity

theta1_3 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
theta2_3 = np.radians(180 + random.uniform(-1, 1))  # Slight random perturbation
omega1_3 = random.uniform(-.5, .5)  # Small random initial velocity
omega2_3 = random.uniform(-.5, .5)  # Small random initial velocity

def doublep(g, l, m, theta1, theta2, omega1, omega2):
# Initial conditions
    
# State vector r = [theta1, theta2, omega1, omega2]
    r0 = np.array([theta1, theta2, omega1, omega2])  

# Time parameters
    dt = 0.01  # Time step
    t_max = 10  # Simulation duration: sets number of TIME STEPS
    t = np.arange(0, t_max, dt)

# Equations of motion for the double pendulum
    def equations(r):
    ## assign the four variables we need to evolve to ONE vector r 
    ## that holds them all
        theta1, theta2, omega1, omega2 = r
        delta_theta = theta2 - theta1

    # Define the four equations for the system
        ftheta1 = omega1
        ftheta2 = omega2

    ## HI    NT: the expressions for fomega1, fomega2 are quite long,
    ## so create smaller expressions to hold the denominators
        denom1 = (3 - np.cos(2 * omega1 - 2 * omega2))
        denom2 = (3 - np.cos(2 * omega1 - 2 * omega2))

        fomega1 = -(omega1**2 * np.sin(2 * omega1 - 2 * omega2) + 2 * omega2**2 * np.sin(omega1 - omega2) + (g / l) * (np.sin(omega1 - 2 * omega2) + 3 * np.sin(theta1))) / denom1

        fomega2 = (4 * omega1**2 * np.sin(omega1 - omega2) + omega2**2 * np.sin(2 * omega1 - 2 * omega2) + 2 * (g / l) * (np.sin(2 * omega1 - omega2) - np.sin(theta2))) / denom2

        return np.array([ftheta1, ftheta2, fomega1, fomega2])

# Runge-Kutta 4th order method
    def rk4_step(r, dt):
        k1 = dt * equations(r)
        k2 = dt * equations(r + k1 / 2)
        k3 = dt * equations(r + k2 / 2)
        k4 = dt * equations(r + k3)
        return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

## this is a CLEVER way to hold all of your data in one object
## R is a vector of lenght t (time steps) that will hold the evolution
## of all FOUR of your variables
## r0 is a VECTOR initialized to r0 = [0,0,0,0]
    R = np.zeros((len(t), 4))
    R[0] = r0

# Integrate equations and save data
## remember: numerical integration --> for loop
    for i in range(1, len(t)):
        R[i] = rk4_step(R[i - 1], dt)

# Extract angles and angular velocities
    theta1_vals, theta2_vals, omega1_vals, omega2_vals = R.T

# Convert to Cartesian coordinates for visualization
    x1 = l * np.sin(theta1_vals)
    y1 = -l * np.cos(theta1_vals)
    x2 = x1 + l * np.sin(theta2_vals)
    y2 = y1 - l * np.cos(theta2_vals)

# Save data
    np.savetxt("double_pendulum_data.txt", np.column_stack([t, x1, y1, x2, y2]),
               header="time x1 y1 x2 y2", comments="")

###-------Animation Time---------###

    data = np.loadtxt("double_pendulum_data.txt", skiprows=1)
    t, x1, y1, x2, y2 = data.T

# Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Perturbed Pendulum Simulation")

# Plot the pivot point (fixed at the origin)
    pivot, = ax.plot([], [], 'ko', label="Pivot")

# Create lines for the pendulum arms
    line1, = ax.plot([], [], 'b-', label="Mass 1 Path")
    line2, = ax.plot([], [], 'r-', label="Mass 2 Path")

# Create markers for the masses
    mass1, = ax.plot([], [], 'bo', label="Mass 1", markersize=8)
    mass2, = ax.plot([], [], 'ro', label="Mass 2", markersize=8)

# Initial conditions for the animation
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        mass1.set_data([], [])
        mass2.set_data([], [])
        return line1, line2, mass1, mass2

# Update function for the animation
    def update(frame):
    # Get the current positions of the masses
        x1_pos = x1[frame]
        y1_pos = y1[frame]
        x2_pos = x2[frame]
        y2_pos = y2[frame]
    
    # Update the data for the lines
        line1.set_data([0, x1_pos], [0, y1_pos])  # Line from pivot to mass 1
        line2.set_data([x1_pos, x2_pos], [y1_pos, y2_pos])  # Line from mass 1 to mass 2

    # Update the positions of the masses
        mass1.set_data(x1_pos, y1_pos)
        mass2.set_data(x2_pos, y2_pos)
    
        return line1, line2, mass1, mass2

# Set up the animation
# Adjust interval and fps
    interval_ms = 10  # 200 ms between frames
    fps = 1000 // interval_ms  # Ensure the fps matches the interval

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval_ms)

## Save the animation as a video (MP4 file)
    ani.save('perturbed_pendulum_simulation.gif', writer='pillow', fps=fps)

    plt.show()

    T = m * l * (omega1**2 + .5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    V = -m * g * l * (2 * np.cos(theta1) + np.cos(theta2))

    Etot = T + V

    print('total energy of the system is', Etot)

    T_vals = m * l * (omega1_vals**2 + .5 * omega2_vals**2 + omega1_vals * omega2_vals * np.cos(theta1_vals - theta2_vals))
    V_vals = -m * g * l * (2 * np.cos(theta1_vals) + np.cos(theta2_vals))
    
    fig, ax = plt.subplots()

    ax.plot(t, T_vals + V_vals)
    
    plt.show()

doublep(g, l, m, theta1_1, theta2_1, omega1_1, omega2_1)
doublep(g, l, m, theta1_2, theta2_2, omega1_2, omega2_2)
doublep(g, l, m, theta1_3, theta2_3, omega1_3, omega2_3)
"""
###------------------------------------------problem 1---------------------------###

##Lab file updated to reflect changes, lab_day18.py

###------------------------------------------problem 2---------------------------###

#doublep(g, l, m, 90, 90, 0, 0)

###------------------------------------------problem 3---------------------------###

fs, audio = wavfile.read('/d/users/iank/Downloads/audio.wav')

compressed_audio = lb.compress_audio_fft(audio)

wavfile.write('audio_compressed.wav', fs, np.real(compressed_audio).astype(np.float32))

###------------------------------------------problem 4---------------------------###

##Jpeg files are created by the use of the fourier transform as a form of compression. The compression, even with high orders of ft, changes the values of the data in a way that even through reverse fourier transform the true values are lost. There are better files available to maintain the sacntity of the data contained.
