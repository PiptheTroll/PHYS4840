##Homework 01
##Ian Karfs

import sys
sys.path.append('/d/users/iank/PHYS4840_labs/')
import libary as lb
import numpy as np

##question 1 done elsewhere##

##question 2##

##lobotomized the advanced cookie solution

def sandwiches(max_money):

	ham_price = 3.65
	apple_price = 4.25
	pbj_price = 3.00
	turkey_price = 3.35

	half_ham_price = round(3.65 * .6, 2)
	half_apple_price = round(4.25 * .6, 2)
	half_pbj_price = round(3.00 * .6, 2)
	half_turkey_price = round(3.35 * .6, 2)

	# List to store all valid combinations
	combinations = []
	spare_change_values=[]

	# Nested loops to consider all possible counts of each sandwich type
	for ham_count in range(int(max_money // ham_price) + 1):
		for apple_count in range(int(max_money // apple_price) + 1):
			for pbj_count in range(int(max_money // pbj_price) + 1):
				for turkey_count in range(int(max_money // turkey_price) + 1):
					for half_ham_count in range(int(max_money // half_ham_price) + 1):
						for half_apple_count in range(int(max_money // half_apple_price) + 1):
							for half_pbj_count in range(int(max_money // half_pbj_price) + 1):
								for half_turkey_count in range(int(max_money // half_turkey_price) + 1):
	                # Calculate the total cost for this combination
									total_cost = (
										ham_count * ham_price +
										apple_count * apple_price +
										pbj_count * pbj_price +
										turkey_count * turkey_price +
										half_ham_count * half_ham_price +
										half_apple_count * half_apple_price +
										half_pbj_count * half_pbj_price +
										half_turkey_count * half_turkey_price
									)
									spare_change = max_money - total_cost
					                # Check if the total cost is within the budget
									if total_cost <= max_money and half_ham_count <= 1 and half_apple_count <= 1 and half_pbj_count <= 1 and half_turkey_count <= 1 and half_ham_count + half_apple_count + half_pbj_count + half_turkey_count >= 1 and spare_change * 4 == int(spare_change * 4) :
										combinations.append(
											(ham_count, apple_count, pbj_count, turkey_count, half_ham_count, half_apple_count, half_pbj_count, half_turkey_count)
										)
										spare_change_values.append(spare_change)

	spare_change_values = np.array(spare_change_values)
	combinations = np.array(combinations)

	#for combo in combinations:
	    #print(f"ham: {combo[0]}, apple: {combo[1]}, pbj: {combo[2]}, turkey: {combo[3]}, half_ham: {combo[4]}, half_apple: {combo[5]}, half_pbj: {combo[6]}, half_turkey: {combo[7]}")

	if len(combinations) == 0:
		print("No possible combinations")
	else:
		least_spare_change_index = spare_change_values.argmin()
		best_sandwich_combo = combinations[least_spare_change_index]
		print('The least spare change is ', "%.2f"%spare_change_values.min())
		print('The best sandwich combo is ', f"ham: {best_sandwich_combo[0]}  "+\
										   f"apple: {best_sandwich_combo[1]}   "+\
										   f"pbj: {best_sandwich_combo[2]}  "+\
										   f"turkey: {best_sandwich_combo[3]} "+\
										   f"half_ham: {best_sandwich_combo[4]}  "+\
										   f"half_apple: {best_sandwich_combo[5]}   "+\
										   f"half_pbj: {best_sandwich_combo[6]}  "+\
										   f"half_turkey: {best_sandwich_combo[7]}"
										   )

sandwiches(10)

##question 3##

def is_prime(n):
	out = True
	for i in range(2, int(np.sqrt(n))+1):
		if n % i == 0:
			out = False
		elif n % i:
			pass
	return out


prime_is = []

for n in range(2, 10000+1):
	if is_prime(n) == True:
		prime_is.append(n)

print(prime_is)

##question 4##
def cat(n):
	if n == 0:
		return 1
	else:
		return (((4 * n) - 2) / (n + 1)) * cat(n - 1)

def g(m,n):
	if n == 0:
		return m
	else:
		return g(n, m % n)

print("C sub 100 is " + str(cat(100)))
print("the greatest common divisor of 108 and 192 is " + str(g(108, 192)))

##question  5##

print('Alan Turing is known for the turing method, the Turing machine, and that movie with Benedict Cumberbatch. The Turing machine is a theoretical machine that can compute as well as any other computing machine. It was intended to show the extent to which computing can be used. The Turing method is a process used to determine whether subject is human or computatonal. The necessity of this process will grow as machine learning technologies improve, I have heard of a small group of language learning models passing short Turing tests. Also the movie is fantastic.')
