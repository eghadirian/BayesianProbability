import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(seed=42)
    count_data = np.random.randint(low=0, high=80, size=100)
    n_count_data = len(count_data)
    with pm.Model() as model:
        alpha = 1.0 / count_data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
        observation = pm.Poisson("obs", lambda_, observed=count_data)
        step = pm.Metropolis()
        trace = pm.sample(10000, tune=5000, step=step)
    lambda_1_samples = trace['lambda_1']
    lambda_2_samples = trace['lambda_2']
    tau_samples = trace['tau']
    plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_1$", color="#A60628", normed=True)
    plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
    plt.show()