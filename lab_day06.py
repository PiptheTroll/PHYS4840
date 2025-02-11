#print(0.1 + 0.2)
#print(repr(0.1 + 0.2))

from math import sqrt

x = 1.0
y = 1.0 + (1e-14)*sqrt(2)

answer_1 = 1e14*(y-x)
answer_2 = sqrt(2)


print("answer1: ", answer_1 )
print("answer2: ", answer_2 ) 

print('percentage error is: ', ((answer_1 / answer_2) * 100) - 100,"%")

###--------------------------------------------------------###

import timeit
import time
import numpy as np
import sys
import pandas as pd

setup_code = "import numpy as np; my_array = np.arange(100000)"

loop_time100 = timeit.timeit("loop = sum([x**2 for x in range(100000)])", setup=setup_code, number=100)
numpy_time100 = timeit.timeit("np.sum(my_array**2)", setup=setup_code, number=100)

loop_time1000 = timeit.timeit("loop = sum([x**2 for x in range(100000)])", setup=setup_code, number=1000)
numpy_time1000 = timeit.timeit("np.sum(my_array**2)", setup=setup_code, number=1000)

print('over 100 iterations, the difference in time is %s'%(numpy_time100 - loop_time100))
print('over 1000 iterations, the difference in time is %s'%(numpy_time1000 - loop_time1000))

###--------------------------------------------###

filename = '/d/users/iank/Downloads/NGC6341.dat'

start_parser = time.perf_counter()

blue, green, red = [], [], []

# Open the file and read line by line
with open(filename, 'r') as file:
    for line in file:
        # Skip lines that start with '#'
        if line.startswith('#'):
            continue
        
        # Split the line into columns based on spaces
        columns = line.split()
        
        blue.append(float(columns[8]))   # Column 9 
        green.append(float(columns[14])) # Column 15 
        red.append(float(columns[26]))   # Column 27 

blue = np.array(blue)
green = np.array(green)
red = np.array(red)


end_parser  = time.perf_counter()

print('Time to run custom parser version: ', end_parser-start_parser, ' seconds')
##.348911 seconds

###-----###

start_pandas = time.perf_counter()

df = pd.read_csv(filename, delim_whitespace=True, comment='#', header=None, skiprows=54)

# Extract the columns corresponding to
# F336W, F438W, and F814W magnitudes
blue = df.iloc[:, 8]   # Column 9 
green = df.iloc[:, 14]  # Column 15 
red = df.iloc[:, 26]   # Column 27 

blue = blue.to_numpy()
green = green.to_numpy()
red = red.to_numpy()

print("len(green):", len(green))

end_pandas  = time.perf_counter()

print('Time to run pandas version: ', end_pandas-start_pandas, ' seconds')
##.705987 seconds