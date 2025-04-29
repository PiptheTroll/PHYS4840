#! /usr/local/Anaconda2022/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

def generate_sinusoidal_data(num_points, amplitude, frequency, noise_std_dev):
    x = np.linspace(0, 10, num_points)
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = sinus_eq(amplitdue, frequency, x) + noise
    return x, y


def sinus_eq(amplitdue, frequency, x):
    phase_shift = np.random.uniform(0, 2 * np.pi)  # Random phase shift
    return amplitude * np.sin(frequency * x + phase_shift)


#make x y data
def generate_linear_data(num_points, gradient, intercept, noise_std_dev):
    rand_nums = np.random.rand(num_points)
    x = 10 * rand_nums
    noise = np.random.normal(0, noise_std_dev, num_points)
    y = linear_eq(x, gradient, intercept) + noise
    return x, y

def linear_eq(X, m, c):
    return m*X + c


# Example usage
num_points = 100  # Number of data points to generate
gradient = 5  # User-defined gradient
intercept = 5  # User-defined intercept
noise_std_dev = 1  # Standard deviation of the noise

X, Y = generate_linear_data(num_points, gradient, intercept, noise_std_dev)



loss_list = []


# Building the model
m = 0.1 #Initial guess for gradient
c = 0 #Initial guess for intercept
L = 0.001  # The learning Rate
epochs = 2000  # The number of iterations to perform gradient descent
n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = linear_eq(X, m, c)  # The current predicted value of Y
    #MSE loss
    loss = abs((-1/n) * np.sum(Y - Y_pred))
    # Derivative wrt m
    Dm = (-1 / n) * np.sum(2 * (Y - (m * X + c))(-X))
    # Derivative wrt c
    Dc = (-2 / n) * np.sum(Y - Y_pred)
    # Update m
    m = m - loss * Dm
    # Update c
    c = c - L * Dc
    loss_list.append(loss)


# Plot the data and the fitted line
plt.scatter(X, Y)
plt.plot(X, m * X + c, color='red')
plt.xlabel('x')
plt.ylabel('y')
# Display fitted m, c, and loss
text = f'm = {m:.3f}\nc = {c:.3f}\nloss = {loss:.3f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top')
# Show the plot
plt.show()


plt.plot(abs(np.array(loss_list)))
plt.xlabel('epoch')
plt.ylabel('lost')
plt.show()




plt.clf()
# Define the range of 'm' and 'c' values
m_values = np.linspace(0,10, 1000)  # Range of 'm' values
c_values = np.linspace(0,10, 1000)  # Range of 'c' values

# Initialize a matrix to store the loss values
loss_values = np.zeros((len(m_values), len(c_values)))

# Compute the loss for each combination of 'm' and 'c' values
for i, m in enumerate(m_values):
    for j, c in enumerate(c_values):
        Y_pred = linear_eq(X, m, c)  # The current predicted value of Y
        loss = abs((-1/n) * np.sum(Y - Y_pred))  # MSE loss)
        loss_values[i, j] = loss


# Plot the heatmap
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(loss_values, cmap='rainbow', extent=[min(c_values), max(c_values), min(m_values), max(m_values)])

# Colorbar
cbar = fig.colorbar(im, ax=ax, label='Loss')

# Plot the circle
circle = plt.Circle((gradient, intercept), radius=0.5, color='blue', fill=False)
ax.add_artist(circle)

# Axis labels and title
ax.set_xlabel('c')
ax.set_ylabel('m')
ax.set_title('Heatmap of Loss')

# Adjust figure size for a square plot
fig.tight_layout(rect=[0, 0, 1, 1])

# Set aspect ratio to make it square
ax.set_aspect('equal')

plt.show()




# Create the grid of 'm' and 'c' values
M, C = np.meshgrid(m_values, c_values)

# Create the figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(C, M, np.log(loss_values), cmap='rainbow')

# Set labels and title
ax.set_xlabel('c')
ax.set_ylabel('m')
ax.set_zlabel('Log Loss')
ax.set_title('3D Plot of Loss')

plt.show()


from scipy.optimize import minimize
import celerite
from celerite import terms
import corner

def generate_sinusoidal_data(n, amp, freq, noise_std, phase=0.0):
    t = np.linspace(0, 10, n)
    y_true = amp * np.sin(2 * np.pi * freq * t + phase)
    y_obs = y_true + np.random.normal(0, noise_std, size=n)
    y_err = np.full(n, noise_std)
    return t, y_obs, y_err

def build_gp(params, t, yerr):
    log_S0, log_Q, log_omega0 = params
    kernel = terms.SHOTerm(log_S0=log_S0, log_Q=log_Q, log_omega0=log_omega0)
    gp = celerite.GP(kernel, mean=0.0)
    gp.compute(t, yerr)
    return gp

def neg_log_likelihood(params, t, y, yerr):
    try:
        gp = build_gp(params, t, yerr)
        return -gp.log_likelihood(y)
    except:
        return 1e25

def main(
    n_points=100,
    amp=2.0,
    period = 5.42069,
    noise_std=.1,
    phase=np.pi/4,
    normalize=True
):

    freq = 1/period

    t, y, yerr = generate_sinusoidal_data(n_points, amp, freq, noise_std, phase)
    idx = np.argsort(t)
    t, y, yerr = t[idx], y[idx], yerr[idx]

    if normalize:
        y = (y - np.mean(y)) / np.std(y)

    init = [np.log(np.var(y)), np.log(1.0), np.log(2 * np.pi * freq)]
    bounds = [
        (np.log(1e-5), np.log(10.0)),
        (np.log(0.5), np.log(100.0)),
        (np.log(0.1), np.log(100.0))
    ]

    result = minimize(neg_log_likelihood, init, args=(t, y, yerr), bounds=bounds, method="L-BFGS-B")
    gp = build_gp(result.x, t, yerr)

    log_S0, log_Q, log_omega0 = result.x
    S0 = np.exp(log_S0)
    Q = np.exp(log_Q)
    omega0 = np.exp(log_omega0)
    period = 2 * np.pi / omega0
    tau = Q / omega0

    print("\nOptimized GP Parameters:")
    print(f"Amplitude (S0)       = {S0:.4f}")
    print(f"Quality factor (Q)   = {Q:.4f}")
    print(f"ω₀ (rad/s)           = {omega0:.4f}")
    print(f"Period (s)           = {period:.4f}")
    print(f"Characteristic time  = {tau:.4f}")

    # Forward prediction range
    t_pred = np.linspace(t.min(), t.max() + period, 1200)
    mu, var = gp.predict(y, t_pred, return_var=True)

    plt.errorbar(t, y, yerr=yerr, fmt=".k", label="Data")
    plt.plot(t_pred, mu, label="GP Prediction")
    plt.fill_between(t_pred, mu - np.sqrt(var), mu + np.sqrt(var), alpha=0.3)
    plt.xlabel("Time")
    plt.ylabel("Flux (normalized)" if normalize else "Flux")
    plt.title(f"GP Fit + Prediction (forward by 1 period ≈ {period:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    samples = np.random.multivariate_normal(result.x, np.eye(3)*0.01, size=1000)
    corner.corner(samples, labels=["log_S0", "log_Q", "log_omega0"], truths=result.x)
    plt.show()

if __name__ == "__main__":
    main()
"""

import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



# -- Model --
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

audio



# -- CONFIG --
SEQ_LEN = 10
PRED_COL = 'DHT_Temperature_C'
INPUT_COLS = [
    'BMP_Temperature_C', 'BMP_Pressure_hPa', 'BMP_Altitude_m',
    'DHT_Humidity_percent', 'BH1750_Light_lx'
]
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 32

# -- LOAD DATA --
df = pd.read_csv("cut_data.csv", parse_dates=['Timestamp'])
df = df.dropna()
df = df.sort_values('Timestamp')
df = df.reset_index(drop=True)

# Normalize
scalers = {}
scaled_df = df.copy()
for col in INPUT_COLS + [PRED_COL]:
    scalers[col] = MinMaxScaler()
    scaled_df[col] = scalers[col].fit_transform(df[[col]])

# -- BUILD SEQUENCES --
X, Y = [], []
for i in range(len(scaled_df) - SEQ_LEN):
    seq_x = scaled_df[INPUT_COLS].iloc[i:i+SEQ_LEN].values
    seq_y = scaled_df[PRED_COL].iloc[i+SEQ_LEN]
    X.append(seq_x)
    Y.append(seq_y)

X = np.array(X)
Y = np.array(Y)

# -- Torch Datasets --
X_torch = torch.tensor(X, dtype=torch.float32)
Y_torch = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)



model = SimpleRNN(input_size=len(INPUT_COLS), hidden_size=HIDDEN_SIZE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -- Training loop --
losses = []
for epoch in range(EPOCHS):
    model.train()
    pred = model(X_torch)
    loss = loss_fn(pred, Y_torch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# -- Loss Plot --
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# -- Prediction (back to real units) --
model.eval()
with torch.no_grad():
    pred = model(X_torch).squeeze().numpy()

real_y = scalers[PRED_COL].inverse_transform(Y.reshape(-1, 1)).squeeze()
pred_y = scalers[PRED_COL].inverse_transform(pred.reshape(-1, 1)).squeeze()

plt.plot(real_y, label="Actual")
plt.plot(pred_y, label="Predicted")
plt.legend()
plt.title("DHT_Temperature_C Prediction")
plt.xlabel("Timestep")
plt.ylabel("Temperature (C)")
plt.tight_layout()
plt.show()

# -- Forecasting into the future (few days) --
# Assume samples are spaced ~5 sec apart -> 17,280 steps ≈ 1 day
samples_per_day = int(86400 / 5)
forecast_days = 3
steps = forecast_days * samples_per_day

seed = X[-1]
future = []

for _ in range(steps):
    x_in = torch.tensor(seed, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_pred = model(x_in).item()

    # De-normalize prediction
    real_pred = scalers[PRED_COL].inverse_transform([[next_pred]])[0][0]
    future.append(real_pred)

    # Build new input window
    new_row = seed[1:].tolist()
    last_input = scaled_df[INPUT_COLS].iloc[-1].tolist()
    new_row.append(last_input)
    seed = np.array(new_row)

# -- Plot forecast
plt.plot(future, label="Future Forecast")
plt.title(f"Forward Prediction ({forecast_days} days)")
plt.xlabel("Timestep (~5s interval)")
plt.ylabel("Temperature (C)")
plt.legend()
plt.tight_layout()
plt.show()
