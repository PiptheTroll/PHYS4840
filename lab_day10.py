import numpy as np
import sys
import libary as lb

x_val = np.arange(-2, 2+1, 100)
h = 10**-10


def cent_diff(x, f, h):
	return ((lb.f1(x + (h / 2))) - (lb.f1(x - (h / 2))) / h)


