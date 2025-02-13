import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def funct_y(x):
 	y = 2.0*x**3.0
 	return y

def cookies(max):

	max_money = float(max)

	sugar_price = 2.65
	choc_price  = 3.20
	snick_price = 3.45
	smore_price = 3.70

	n_sugar = np.floor(max_money / sugar_price)

	n_choc  = np.floor(max_money / choc_price)
	n_snick = np.floor(max_money / snick_price)
	n_smore = np.floor(max_money / smore_price)

	change_sugar = max_money % sugar_price
	change_choc  = max_money % choc_price
	change_snick = max_money % snick_price
	change_smore = max_money % smore_price
	
	change_sugar = round(change_sugar, 2)
	change_choc  = round(change_choc, 2)
	change_snick = round(change_snick, 2)
	change_smore = round(change_smore,2)

	print('you can have', n_sugar, ' sugar cookies with $',change_sugar, ' in change remaining')
	print('you can have', n_choc, ' chocolate cookies with $',change_choc, ' in change remaining')
	print('you can have', n_snick, ' snickerdoodles with $',change_snick, ' in change remaining')
	print('you can have', n_smore, " s'mores cookies with $",change_smore, ' in change remaining')

	cookie_type  = np.array(['sugar', 'chocolate', 'snickerdoodle', "s'mores"])
	spare_change = np.array([change_sugar, change_choc, change_snick, change_smore])

	print('minimum spare change: ', spare_change.min())
	print('index of the minimum value in the array: ', spare_change.argmin() )


	which_cookie = cookie_type[spare_change.argmin()]
	print('The single type of cookie that minimizes change is: ', which_cookie)

def gaussian(x, A, B, C, D, E):
#A,B,C,D,E are the changeable paramaters, use an array for the x values.
	return A + (B * x) + (C * (np.exp(-(x - D)**2 / (2 * E**2))))


