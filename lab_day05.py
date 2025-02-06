import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb

blue, green, red = np.loadtxt('/d/users/iank/Downloads/NGC6341.dat', usecols=(8, 14, 26), unpack=True)

quality_cut = np.where((red > 2) &\
						(blue > 10) &\
						(green > 10)
					)

magnitude = (blue)
color = (blue - red)

acceptable_color = color[quality_cut]
acceptable_magnitude = magnitude[quality_cut]

fig, ax  = plt.subplots(figsize=(5, 8))
ax.scatter(acceptable_color, acceptable_magnitude, s=.05, color='black')
ax.invert_yaxis()
ax.set(xlim=(-2,5), ylim=(25,14))

plt.show()
fig.savefig('close_enough.png')