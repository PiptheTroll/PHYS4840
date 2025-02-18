import numpy as np
import matplotlib.pyplot as plt
import time

# Example usage with array data
def trapezoidal(y_values, x_values, N):
    """
    Approximates the integral using trapezoidal rule for given y_values at given x_values.
    
    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals.

    Returns:
        float: The approximated integral.
    """
    a = 0
    b = 1
    h = (b - a) / N

    integral = (1/2) * (y_values[0] + y_values[-1]) * h  # First and last terms

    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly
        yk = np.interp(xk, x_values, y_values)  # Interpolate y at x_k manually in loop
        integral += yk * h

    return integral


# Simpson's rule for array data
def simpsons(y_values, x_values, N):
    """
    Approximates the integral using Simpson's rule for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """

    a = 0
    b = 1
    h = (b - a) / N

    integral = y_data[0] + y_data[-1]

    for k in range(1, N, 2):  # Odd indices (weight 4)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 4 * yk

    for k in range(2, N, 2):  # Even indices (weight 2)
        xk = a + k * h
        yk = np.interp(xk, x_values, y_values)
        integral += 2 * yk

    return (h / 3) * integral  # Final scaling


# Romberg integration for array data
def romberg(y_values, x_values, max_order):
    """
    Approximates the integral using Romberg's method for given y_values at given x_values.

    Parameters:
        y_values (array-like): The function values at given x points.
        x_values (array-like): The x values corresponding to y_values.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))
    a = 0
    b = 1
    N = 0
    h = (b - a)

    # First trapezoidal estimate
    R[0, 0] = (h / 2) * (y_values[0] + y_values[-1])

    for i in range(1, max_order):
        N = 2**i #Remember: we are recomputing the integral with different N (and therefore h)
        h = (b - a) / 2**i #Look at the github derivation for richardson extrapolation

        sum_new_points = sum(np.interp(a + k * h, x_values, y_values) for k in range(1, N, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_new_points

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]


def timing_function(integration_method, x_values, y_values, integral_arg):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        x_values (array-like): The x values.
        y_values (array-like): The corresponding y values.
        integral_arg (int, optional): EITHER Number of intervals to use (Simpson/Trapz) OR the maximum order of extrapolation (Romberg).

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(y_values, x_values, integral_arg)
    end_time = time.perf_counter()
    
    return end_time - start_time, result



# Function to integrate
def function(x):
    return x * np.exp(-x)

# Precompute data for fair comparisons
x_data = np.linspace(0, 1, 100000000)  # High-resolution x values
y_data = function(x_data)

# Testing parameters
N = 1# Number of intervals
max_order = 2# Romberg's accuracy level

# Measure timing for custom methods
trap_time, trap_result = timing_function(trapezoidal, x_data, y_data, N)
simp_time, simp_result = timing_function(simpsons, x_data, y_data, N)
romb_time, romb_result = timing_function(romberg, x_data, y_data, max_order)

# True integral valuetime.perf_counter()
true_value = 0.26424111765711535680895245967707826510837773793646433098432639660507700851

# Compute errors
trap_error = np.abs(trapezoidal(y_data, x_data, N) - true_value)
simp_error = np.abs(simpsons(y_data, x_data, N) - true_value)
romb_error = np.abs(romberg(y_data, x_data, max_order) - true_value)

# Print results with error analysis
print("\nIntegration Method Comparison")
print("=" * 80) # why 80? https://peps.python.org/pep-0008/
print(f"{'Method':<25}{'Result':<20}{'Error':<20}{'Time (sec)':<15}")
print("-" * 80)
print(f"{'Custom Trapezoidal':<25}{trap_result:<20.8f}{trap_error:<20.8e}{trap_time:<15.6f}")
print(f"{'Custom Simpsons':<25}{simp_result:<20.8f}{simp_error:<20.8e}{simp_time:<15.6f}")
print(f"{'Custom Romberg':<25}{romb_result:<20.8f}{romb_error:<20.8e}{romb_time:<15.6f}")
print("=" * 80)

trap_times = []
simp_times = []
romb_times = []

trap_errors = []
simp_errors = []
romb_errors = []

Num = np.arange(1, 15)


for n in Num:
	trap_times.append(timing_function(trapezoidal, x_data, y_data, n)[0])
	simp_times.append(timing_function(simpsons, x_data, y_data, n)[0])
	romb_times.append(timing_function(romberg, x_data, y_data, n)[0])

	trap_errors.append(np.abs(trapezoidal(y_data, x_data, n) - true_value))
	simp_errors.append(np.abs(simpsons(y_data, x_data, n) - true_value))
	romb_errors.append(np.abs(romberg(y_data, x_data, n) - true_value))

trap_times = np.array(trap_times)
simp_times = np.array(simp_times)
romb_times = np.array(romb_times)
trap_errors = np.array(trap_errors)
simp_errors = np.array(simp_errors)
romb_errors = np.array(romb_errors)

fig, ax = plt.subplots(2)

ax[0].plot(Num, trap_errors, color ='black')
ax[0].plot(Num, simp_errors, color ='blue')
ax[0].plot(Num, romb_errors, color = 'green')
ax[1].plot(np.log10(trap_times), trap_errors, color ='black')
ax[1].plot(np.log10(simp_times), simp_errors, color ='blue')
ax[1].plot(np.log10(romb_times), romb_errors, color = 'green')

plt.show()