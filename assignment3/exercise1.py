import numpy as np

np.random.seed(0)
N = 10000

def estimate(x1, x2):   
    r = 0.5
    inside = ((x1 - r)**2 + (x2 - r)**2) <= r**2
    N_in = inside.sum()
    X_hat = N_in / len(x1)
    pi_hat = 4 * X_hat
    return N_in, X_hat, pi_hat

# Exercise 1.3: uniform samples
x1 = np.random.uniform(0, 1, N)
x2 = np.random.uniform(0, 1, N)
Nin1, X1, pi1 = estimate(x1, x2)
print(f"Uniform samples: N_in={Nin1}, X_hat={X1:.4f}, pi_hat={pi1:.4f}")

# Exercise 1.6: correlated samples
x1_corr = np.random.uniform(0, 1, N)
eps = np.random.uniform(-0.1, 0.1, N)
x2_corr = x1_corr + eps
Nin2, X2, pi2 = estimate(x1_corr, x2_corr)
print(f"Correlated samples: N_in={Nin2}, X_hat={X2:.4f}, pi_hat={pi2:.4f}")
