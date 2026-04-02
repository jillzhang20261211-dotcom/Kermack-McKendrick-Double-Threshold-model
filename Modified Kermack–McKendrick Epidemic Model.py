#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

m = 3.0
R0_values = [0.5, 1.0, 1.5, 2.0]

def f(y):
    return 1 - np.exp(-y)

def RHS(J, J0, R0):
    return R0 * f(max(J - m, 0)) + J0

def solve_fixed_point(J0, R0, tol=1e-6):
    J = J0
    for _ in range(1000):
        J_new = RHS(J, J0, R0)
        if abs(J_new - J) < tol:
            return J_new
        J = J_new
    return J

J0_values = np.linspace(0, 7, 300)

colors = {
    0.5: 'red',
    1.0: 'blue',
    1.5: 'orange',
    2.0: 'green'
}

plt.figure(figsize=(7,5))

for R0 in R0_values:
    J_inf_values = []
    for J0 in J0_values:
        J_inf = solve_fixed_point(J0, R0)
        J_inf_values.append(J_inf)
    
    plt.plot(J0_values, J_inf_values, linewidth=2.5, color=colors[R0], label=fr"$R_0 = {R0:.2f}$")

plt.axvline(x=m, linestyle=':', color='gray', linewidth=1.5)

plt.xlabel(r"$J_0^\infty$")
plt.ylabel(r"$J^\infty$")
plt.title("Final Epidemic Size as a Function of Final Cumulative Primary Force of Infection (Fixed Resistance)")
plt.legend(frameon=False)
plt.ylim(0, 7)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

R0_values = [2.2, 4.5, 5.0, 5.5]

shape = 3.0
scale = 1.0

def sigma(m):
    return gamma.pdf(m, a=shape, scale=scale)

def f(y):
    return 1 - np.exp(-y)

def integral_term(J):
    m_vals = np.linspace(0, J, 1000)
    integrand = sigma(m_vals) * f(J - m_vals)
    return np.trapz(integrand, m_vals)

def solve_gamma(J0, R0):
    J = J0
    for _ in range(2000):
        J_new = R0 * integral_term(J) + J0
        if abs(J_new - J) < 1e-7:
            return J_new
        J = J_new
    return J

J0_values = np.linspace(0, 4, 400)

colors = {
    2.2: 'red',
    4.5: 'blue',
    5.0: 'orange',
    5.5: 'green'
}

plt.figure(figsize=(7,5))

for R0 in R0_values:
    J_inf = []
    for J0 in J0_values:
        J_inf.append(solve_gamma(J0, R0))
    
    plt.plot(J0_values, J_inf, linewidth=2.5, color=colors[R0], label=fr"$R_0 = {R0:.2f}$")

plt.xlabel(r"$J_0^\infty$")
plt.ylabel(r"$J^\infty$")
plt.title("Final Epidemic Size as a Function of Final Cumulative Primary Force of Infection (Gamma-Distributed)")
plt.legend(frameon=False)
plt.ylim(0, 7)

plt.show()

