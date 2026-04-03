#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

T = 1.0
N = 1000
dt = T / N
mu = 0.05
sigma = 0.2
beta = 2.0
gamma = 1.0
eta = 0.3
alpha = 0.8
kappa = 0.01
m = 1.0
delta = 3.0

S = np.zeros(N)
S[0] = 100
I = np.zeros(N)
I[0] = 0.1
J = np.zeros(N)

def f(x):
    return 1 - np.exp(-x)

for t in range(1, N):
    dW1 = np.sqrt(dt) * np.random.randn()
    dW2 = np.sqrt(dt) * np.random.randn()
    J[t] = J[t-1] + dt * (I[t-1] - delta * J[t-1])
    infection_force = beta * f(max(J[t-1] - m, 0))
    I[t] = I[t-1] + (infection_force - gamma * I[t-1]) * dt \
           + eta * np.sqrt(max(I[t-1], 0)) * dW2
    I[t] = max(I[t], 0)
    contagion = f(max(J[t] - m, 0))
    impact = alpha * contagion / (1 + kappa * S[t-1])
    S[t] = S[t-1] * (1 + mu*dt + sigma*dW1 + impact * dt)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(S)
plt.title("Price Dynamics")

plt.subplot(1,2,2)
plt.plot(J)
plt.title("Effective Exposure")
plt.tight_layout()
plt.show()
S_gbm = np.zeros(N)
S_gbm[0] = 100
for t in range(1, N):
    dW = np.sqrt(dt) * np.random.randn()
    S_gbm[t] = S_gbm[t-1] * (1 + mu*dt + sigma*dW)
plt.plot(S, label="Model")
plt.plot(S_gbm, label="GBM")
plt.legend()
plt.show()
