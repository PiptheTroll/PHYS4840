import numpy as np
import matplotlib.pyplot as plt
import time
import os

"""
# Domain setup
N = 50
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
f = np.cos(x)  # Function to integrate

# Manual computation of Fourier coefficients: f-hat
def manual_fourier_coeffs(f, x):
    N = len(x)
    dx = x[1] - x[0]
    L = N * dx  # Total domain length

    # Manually construct frequency indices k
    # Assume domain is periodic over [0, L)
    
    ## fft.fftfreq
    k = np.zeros(N)
    for n in range(N):
        if n <= N // 2:
            k[n] = n
        else:
            k[n] = n - N
    k = (2 * np.pi / L) * k  # Convert to angular frequencies
    ## fft.fftfreq

    # Compute Fourier coefficients manually
    fhats = np.zeros(N, dtype=complex)
    for n in range(N):
        exponent = -1j * k[n] * x
        fhats[n] = (1 / N) * np.sum(f * np.exp(exponent))

    return k, fhats


k, fk = manual_fourier_coeffs(f, x)

for i in range(len(k)):
    print("coeff number", i,\
          "    Fourier wave number k:", int(k[i]),\
          "    Fourier coefficient fk:", fk[i])

# Integrate: divide by i*k (except at k=0)
Fk = np.zeros_like(fk) ## populate an object the shape of fk with zeros 
nonzero = k != 0 # the set 'nonzero' is defined as those for which k != 0 (slicker version of np.where() )
Fk[nonzero] = fk[nonzero] / (1j * k[nonzero]) # perform the integration, AKA division in Fourier, for k[nonzero]
Fk[~nonzero] = 0.0  # ~nonzero means "not nonzero," so we are separating the 0-entry of FK out 
                    # and setting it to zero so as not to enounter a divide-by-zero error

# Reconstruct integrated function

## fft.ifft
f_integrated = np.real(np.sum([Fk[n] * np.exp(1j * k[n] * x) for n in range(N)], axis=0))
## fft.ifft

# Plot
x_array_for_sin = np.linspace(-np.pi, np.pi, 10000)

plt.plot(x_array_for_sin, np.sin(x_array_for_sin), '-', linewidth=1, color='black', label='True sin(x)')
plt.plot(x, f_integrated, "go-", linewidth=2, markersize=10, alpha=0.6, label='Integrated cos(x)')

plt.legend()
plt.title("Spectral Integration via Manual Fourier Series")
plt.grid(True)
plt.show()

###-------Numpy Method-------###

N = 50                             
L = 2 * np.pi                      # Domain length [-π, π]
x = np.linspace(-np.pi, np.pi, N, endpoint=False)  # Grid points
f = np.cos(x)                      # Function to integrate

# Step 2: Compute the FFT of the function (go to frequency space)
f_hat = np.fft.fft(f)              # Fourier coefficients (complex)

# Step 3: Build the frequency array
k = np.fft.fftfreq(N, d=L/N)       # Frequencies in cycles per unit length
k = 2 * np.pi * k                  # Convert to angular frequencies (radians)

# Step 4: Integrate in Fourier space by dividing by ik
F_hat = np.zeros(N, dtype=complex) # Initialize integrated coefficients

for i in range(N):
    if k[i] != 0:
        F_hat[i] = f_hat[i] / (1j * k[i])
    else:
        F_hat[i] = 0  # No contribution from the DC component (mean value)

# Step 5: Inverse FFT to return to real space
f_integrated = np.fft.ifft(F_hat).real  # Take the real part

# Plot results
x_array_for_sin = np.linspace(-np.pi, np.pi, 10000)

plt.plot(x_array_for_sin, np.sin(x_array_for_sin), '-', linewidth=1, color='black', label='True sin(x)')
plt.plot(x, f_integrated, "ro-", linewidth=2, markersize=10, alpha=0.6, label='Integrated cos(x)')

plt.legend()
plt.title("Spectral Integration using FFT")
plt.grid(True)
plt.show()


seed = 69

# Mersenne Twister (default legacy)
rng1 = np.random.default_rng(np.random.MT19937(seed))
print("Mersenne Twister:", rng1.random(1))

# PCG64 (NumPy default)
rng2 = np.random.default_rng(seed)
print("PCG64:", rng2.random(1))

# Philox (counter-based PRNG, parallel-safe)
rng3 = np.random.Generator(np.random.Philox(seed))
print("Philox:", rng3.random(1))

# SFC64 (good for speed and statistical quality)
rng4 = np.random.Generator(np.random.SFC64(seed))
print("SFC64:", rng4.random(1))

time1 = time.perf_counter()

from math import sqrt,log,cos,sin,pi
from random import random

Z = 79
e = 1.602e-19
E = 7.7e6*e
epsilon0 = 8.854e-12
a0 = 5.292e-11
sigma = a0/100

N = int(1e7)

def gaussian():
## does the process in fortran on lines 27-33, creates random values using the gaussian
    r = sqrt(-2*sigma*sigma*log(1-random()))
    theta = 2*pi*random()
    x = r*cos(theta)
    y = r*sin(theta)
    return x,y

count = 0
for i in range(N):
##the next two lines actually run the code in lines 26-34 in the fortran code.
    x,y = gaussian()
    b = sqrt(x*x+y*y)
## sets the same restriction for escape as line 36 in the fortran code
    if b<Z*e*e/(2*pi*epsilon0*E):
        count += 1

print(count,"particles were reflected out of",N)
"""
time2 = time.perf_counter()

os.system('gfortran oddball.f90 -o oddball3.exe')
os.system('./oddball3.exe')

time3 = time.perf_counter()

#print('python time is', time2 - time1)
print('fortran time is', time3 - time2)
