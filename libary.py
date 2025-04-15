import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import time

def gaussian(x, A, B, C, D, E):
#A,B,C,D,E are the changeable paramaters, use an array for the x values.
	return A + (B * x) + (C * (np.exp(-(x - D)**2 / (2 * E**2))))

def format_axes(ax):
##creates a more visually appealing axis design
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)  # Larger major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=1)    # Minor ticks
    ax.minorticks_on()  # Enable minor ticks

def distance_mod(p):
##calculates the distance modulus (m - M) with p in parsecs 
	return 5 * np.log10(p/10)

def f1(x, t):
## 1 + tanh(2x)/2 function used starting in lab_day10
    return 1 + (np.tanh(2 * x) / 2)

def f2(x, t):
## derivative of the f1 functions as a function
    return 1 / np.cosh(2 * x)**2

def f3(x, t):
    return x**2 - x

def f4(x, t):
    return -x**3 + np.sin(t)

def euler_method(f, x0, t0, t_end, dt):
#solves a FO ODE f, with initial values x0 at time t0, up until the time t_end, with step sizes (h) of size dt
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + dt * f(x_values[i - 1], t_values[i - 1])

    return t_values, x_values

def RK(f, x0, t0, t_end, dt):
#Runge-Kutta method for ODE DEQ solving
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + dt * f(x_values[i - 1] + .5 * (dt * f(x_values[i - 1], t_values[i - 1])), t_values[i - 1] + .5 * dt)

    return t_values, x_values

def RK4(f, x0, t0, t_end, dt):
#Runge-Kutta method for ODE DEQ solving to the fourth order
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
#sets the initial value of the x_values to the x0 initial value
    def k1(x, t, f, dt):
        return dt * f(x, t)
    def k2(x, t, f, dt):
        return dt * f(x + k1(x, t, f, dt) / 2, t + dt / 2)
    def k3(x, t, f, dt):
        return dt * f(x + k2(x, t, f, dt) / 2, t + dt / 2)
    def k4(x, t, f, dt):
        return dt * f(x + k3(x, t, f, dt), t + dt)

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i - 1] + (k1(x_values[i - 1], t_values[i - 1], f, dt) + k2(x_values[i - 1], t_values[i - 1], f, dt) + k3(x_values[i - 1], t_values[i - 1], f, dt) + k4(x_values[i - 1], t_values[i - 1], f, dt)) / 6

    return t_values, x_values

def compute_a0(func, period=2*np.pi, num_points=1000):
    """
    Compute the a0 Fourier coefficient (constant term).
    
    Parameters:
        func (callable): Function to approximate
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: a0 coefficient (divided by 2)
    """
    x = np.linspace(0, period, num_points)
    y = func(x)

    result = np.trapz(y, x)#Ingegral of this
    return (1 / period) * result


def compute_an(func, n, period=2*np.pi, num_points=1000):
    """
    Compute the an Fourier coefficient for cosine terms using NumPy's trapz.
    
    Parameters:
        func (callable): Function to approximate
        n (int): Harmonic number
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: an coefficient
    """
    x = np.linspace(0, period, num_points)
    y = func(x)
    
    # Create the integrand: f(x) * cos(2*pi*n*x/period)
    integrand = (y * np.cos(2 * np.pi * n * x / period))
    
    result = np.trapz(integrand, x)#integral of integrand
    
    # Scale by 2/period for the Fourier coefficient
    return (2/period) * result



def compute_bn(func, n, period=2*np.pi, num_points=1000):
    """
    Compute the bn Fourier coefficient for sine terms using NumPy's trapz.
    
    Parameters:
        func (callable): Function to approximate
        n (int): Harmonic number
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        float: bn coefficient
    """
    x = np.linspace(0, period, num_points)
    y = func(x)
    
    # Create the integrand: f(x) * sin(2*pi*n*x/period)
    integrand = (y * np.sin(2 * np.pi * n * x / period))
    
    # Use NumPy's trapz function
    result = np.trapz(integrand, x)#see the other gh
    
    # Scale by 2/period for the Fourier coefficient
    return (2/period) * result

def compute_coefficients(func, n_terms, period=2*np.pi, num_points=1000):
    """
    Compute all Fourier coefficients up to a specified number of terms.
    
    Parameters:
        func (callable): Function to approximate
        n_terms (int): Number of terms in the Fourier series
        period (float): Period of the function
        num_points (int): Number of points for numerical integration
    
    Returns:
        tuple: (a0, an_coefficients, bn_coefficients)
    """
    a0 = compute_a0(func, period, num_points)
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)
    
    for n in range(1, n_terms + 1):
        an[n-1] = compute_an(func, n, period, num_points)
        bn[n-1] = compute_bn(func, n, period, num_points)
    
    return a0, an, bn


def fourier_series_approximation(x, a0, an, bn, period=2*np.pi):
    """
    Compute the Fourier series approximation using precomputed coefficients.
    
    Parameters:
        x (array): Points where to evaluate the approximation
        a0 (float): Constant coefficient (divided by 2)
        an (array): Cosine coefficients
        bn (array): Sine coefficients
        period (float): Period of the function
    
    Returns:
        array: Fourier series approximation at points x
    """
    result = np.ones_like(x) * a0
    
    for n in range(1, len(an) + 1):
        result += an[n-1] * np.cos(2 * np.pi * n * x / period)
        result += bn[n-1] * np.sin(2 * np.pi * n * x / period)
    
    return result

#typically we dont need this function but its good for visualising
def compute_partial_approximations(x, a0, an, bn, period=2*np.pi):
    """
    Compute partial Fourier approximations with increasing number of terms.
    
    Parameters:
        x (array): Points where to evaluate the approximation
        a0 (float): Constant coefficient (divided by 2)
        an (array): Cosine coefficients
        bn (array): Sine coefficients
        period (float): Period of the function
    
    Returns:
        list: Approximations with increasing number of terms
    """
    approximations = []
    times = []
    for k in range(1, len(an) + 1):
        time1 = time.perf_counter()
        approx = np.ones_like(x) * a0
        for n in range(k):
            approx += an[n] * np.cos(2 * np.pi * (n+1) * x / period)
            approx += bn[n] * np.sin(2 * np.pi * (n+1) * x / period)
        time2 = time.perf_counter()
        times.append(time2 - time1)
        approximations.append(approx)

    
    return approximations, times

def dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of the input signal.
    
    Parameters:
        x (array): Input signal (time domain)
    
    Returns:
        array: Fourier Transform of x (frequency domain, complex values)
    """
    time1 = time.perf_counter()
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    time2 = time.perf_counter()
    Time = time2 - time1
    return X, Time

def idft(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of the input spectrum.
    
    Parameters:
        X (array): Input spectrum (frequency domain)
    
    Returns:
        array: Inverse Fourier Transform of X (time domain)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    
    # Normalize by N
    x = x / N
    
    return x

def fft_bluestein(x):
    time1 = time.perf_counter()
    N = len(x)
    M = 2**int(np.ceil(np.log2(2*N - 1)))  # Next power of 2 >= 2N - 1
    a = np.array(x, dtype=complex)

    # Chirp signal
    n = np.arange(N)
    chirp = np.exp(1j * np.pi * (n**2) / N)
    
    a_chirp = a * chirp
    b = np.zeros(M, dtype=complex)
    b[:N] = np.exp(-1j * np.pi * (n**2) / N)
    b[-(N-1):] = np.exp(-1j * np.pi * (n[1:][::-1]**2) / N)

    A = np.fft.fft(a_chirp, n=M)
    B = np.fft.fft(b, n=M)
    C = A * B
    c = np.fft.ifft(C)[:N]

    f_inal = c * chirp
    time2 = time.perf_counter()
    Time = time2 - time1
    return f_inal, Time

def fft_zeropad(x):
    time1 = time.perf_counter()
    N = len(x)
    next_pow2 = 1 << (N - 1).bit_length()
    x_padded = np.pad(x, (0, next_pow2 - N), mode='constant')

    f_inal = fft_ct(x_padded)
    time2 = time.perf_counter()
    Time = (time2 - time1)
    return f_inal, Time

def fft_ct(x):
    """
    Compute the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.
    This implementation works for signal lengths that are powers of 2.
    
    Parameters:
        x (array): Input signal (time domain)
    
    Returns:
        array: Fourier Transform of x (frequency domain)
    """
    time1 = time.perf_counter()
    N = len(x)
    
    #print(x)
    #print(N)

    # Base case: FFT of a single point is the point itself
    if N == 1:
        return x
    
    # Check if N is a power of 2
    if N & (N - 1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Split even and odd indices
    even = np.fft.fft(x[0::2])
    odd = np.fft.fft(x[1::2])
    
    # Twiddle factors
    twiddle = np.exp(-2j * np.pi * np.arange(N//2) / N)
    
    # Combine using butterfly pattern
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    for k in range(half_N):
        result[k] = even[k] + twiddle[k] * odd[k]
        result[k + half_N] = even[k] - twiddle[k] * odd[k]
    
    time2 = time.perf_counter()

    Time = (time2 - time1)

    return result, Time

def ifft(X):
    """
    Compute the Inverse Fast Fourier Transform (IFFT).
    
    Parameters:
        X (array): Input spectrum (frequency domain)
    
    Returns:
        array: Inverse Fourier Transform of X (time domain)
    """
    N = len(X)
    
    # Compute the FFT of the conjugate, then conjugate the result and scale
    x = np.conj(np.fft.fft(np.conj(X))) / N
    
    return x


def compress_audio_fft(audio, keep_ratio=0.1):
    """
    Compress audio by keeping only the top `keep_ratio` frequency components (by magnitude).
    
    Parameters:
        audio (np.ndarray): Time-domain audio signal
        keep_ratio (float): Fraction of strongest frequencies to keep (0 < keep_ratio <= 1)
        
    Returns:
        compressed_audio (np.ndarray): Reconstructed audio from compressed frequency domain
        X_compressed (np.ndarray): The compressed spectrum (mostly zero)
    """
    N = len(audio)
    X = np.fft.fft(audio)
    magnitudes = np.abs(X)
    

    # Determine how many frequencies to keep
    new_N = N * keep_ratio 

    # Get indices of top frequencies by magnitude
    #can we assume its already sorted? np.argsort() might be useful here...
    freqnew = np.argsort(X)
    # Create a compressed version of the spectrum
    x_compressed = freqnew[-N:]

    # Inverse FFT to get time-domain signal
    #remember we did all of this on the frequency domain, turn it back into a signal...
    new_audio = ifft(x_compressed)

    return new_audio

def laplacian_operator(Phi, dx, dy, dz):
    """
    Compute the Laplacian of a scalar field Phi (i.e., apply the Poisson operator)
    using central finite differences on a 3D uniform grid.

    Parameters:
    - Phi : 3D numpy array of shape (nx, ny, nz)
    - dx, dy, dz : grid spacings in x, y, z directions

    Returns:
    - laplacian : 3D numpy array of the same shape as Phi
    """

    laplacian = (
        (np.roll(Phi, -1, axis=0) - 2*Phi + np.roll(Phi, 1, axis=0)) / dx**2 +
        (np.roll(Phi, -1, axis=1) - 2*Phi + np.roll(Phi, 1, axis=1)) / dy**2 +
        (np.roll(Phi, -1, axis=2) - 2*Phi + np.roll(Phi, 1, axis=2)) / dz**2
    )

    return laplacian
