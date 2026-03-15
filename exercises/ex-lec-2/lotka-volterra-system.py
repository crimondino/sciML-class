#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#%%
# Define the Predator-Prey system
def lotka_volterra(t, state, alpha, beta, delta, gamma):
    prey, pred = state
    d_prey = alpha * prey - beta * prey * pred
    d_pred = delta * prey * pred - gamma * pred
    return [d_prey, d_pred]

#%%
# Parameters 
# alpha, beta, delta, gamma
params = (0.7, 1.3, 1.1, 0.9)
initial_state = [1.0, 1.0]  # 1 rabbit, 1 fox, equal density
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000) # High resolution for smooth curves

# Solve the system
sol = solve_ivp(lotka_volterra, t_span, initial_state, args=params, t_eval=t_eval)

# Convert to a Pandas DataFrame (Great for Scikit-Learn later)
df = pd.DataFrame({
    'time': sol.t,
    'prey': sol.y[0],
    'predator': sol.y[1]
})

#%%
# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Time Series
ax1.plot(df['time'], df['prey'], label='Prey (Rabbits)', color='skyblue', lw=2)
ax1.plot(df['time'], df['predator'], label='Predators (Foxes)', color='salmon', lw=2)
ax1.set_title("Population Time Series")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population Size")
ax1.legend()

# Plot Phase Space (Predator vs Prey)
ax2.plot(df['prey'], df['predator'], color='purple')
ax2.set_title("Phase Portrait (Stability Orbit)")
ax2.set_xlabel("Prey Population")
ax2.set_ylabel("Predator Population")

plt.tight_layout()
plt.show()
# %%
# Compute the time derivative of the synthetic data
df['dprey'] = np.zeros(len(df))
df['dpredator'] = np.zeros(len(df))

for i in range(1, len(df)-1):
    df.loc[i, 'dprey'] = (df.loc[i+1,'prey'] - df.loc[i-1, 'prey'])/(df.loc[i+1, 'time'] - df.loc[i-1, 'time'])
    df.loc[i, 'dpredator'] = (df.loc[i+1,'predator'] - df.loc[i-1,'predator'])/(df.loc[i+1, 'time'] - df.loc[i-1, 'time'])
# %%
Amatrix = np.array