import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb

###--------question 0--------------------------###

blue, green, red, p = np.loadtxt('/d/users/iank/Downloads/NGC6341.dat', usecols=(8, 14, 26, 32), unpack=True)

quality_cut = np.where((red > 2) &\
						(blue > 10) &\
						(green > 10) &\
						(p > 0 )
					)

magnitude = (blue)
color = (blue - red)

acceptable_color = color[quality_cut]
acceptable_magnitude = magnitude[quality_cut]
prob = p[quality_cut]

fig, ax  = plt.subplots(figsize=(5, 8))
ax.scatter(acceptable_color, acceptable_magnitude, s=.05, c=prob)
ax.invert_yaxis()
ax.set(xlim=(-2,5), ylim=(25,14))

plt.show()

###--------question 1--------------------------###

blueHST, greenHST, redHST = np.loadtxt('/d/users/iank/Downloads/NGC6341.dat', usecols=(8, 14, 26), unpack=True)
blueMIST, greenMIST, redMIST = np.loadtxt('/d/users/iank/Downloads/MIST_v1.2_feh_m1.75_afe_p0.0_vvcrit0.4_HST_WFPC2.iso.cmd', usecols=(12, 14, 20), unpack=True)

def f_2_m(x):
	return 2.5*np.log10(x)

fig, ax  = plt.subplots()
ax.scatter((redHST), blueHST, s=.1, color='black')
ax.scatter((-blueMIST) + 20, (-redMIST) + 22, s=.1, color='green')
ax.set_xlim(32, 5)
ax.set_ylim(30, 10)
plt.show()

## this isochorone model is not a good fit for the data provided. 

###--------question 2--------------------------###

def y(x):
	return x**4

x = np.arange(-100, 101)

fig, ax = plt.subplots(1, 3)
ax[0].plot(x, y(x), color='blue')
ax[0].set_title('linear scale')
ax[0].grid()
ax[0].set_xlabel('x linear')
ax[0].set_ylabel('y linear')
ax[1].plot(x, y(x), color='orange')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_title('log scale')
ax[1].grid()
ax[1].set_xlabel('log(x)')
ax[1].set_ylabel('log(y)')
ax[2].plot(x, np.log10(y(x)), color='green')
ax[2].set_title('log of function')
ax[2].grid()
ax[2].set_xlabel('x linear')
ax[2].set_ylabel('y linear')

plt.show()

###--------question 3--------------------------###

time, spots = np.loadtxt('/d/users/iank/Downloads/sunspots.txt', usecols=(0, 1), unpack=True)

###part a

plt.plot(time, spots, linewidth='.5', color='black')
plt.show()

###part b

plt.plot(time[0:1001], spots[0:1001], linewidth='.5', color='black')
plt.show()

###part c

spots1000 = spots[0:1001]

run_avg = []
for i in range(len(spots1000) - (5) + 1):
    window = spots1000[i : i + (5)]
    window_average = sum(window) / (5)
    run_avg.append(window_average)

fig, ax = plt.subplots()
ax.plot(time[0:997], run_avg, linewidth='.8', color='black')
ax.plot(time[0:1001], spots[0:1001], linewidth='.8', color='orange')
plt.show()

###--------question 4--------------------------###
###part a

#git pull
#git add 'hw3.py'
#git commit -m 'my finished work for the third homework assignment'
#git add 'libary.py'
#git commit -m 'changes made to my code library'
#git push

###part b

#rm -rf .git
#rm -rf .gitignore

###--------question 5--------------------------###
