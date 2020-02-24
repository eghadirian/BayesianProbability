# Gaussian Mixture Model
import numpy as np 
import matplotlib.pyplot as plt 
import pymc3 as pm 

# mock data
np.random.seed(42)
n_data = 1000
weight = np.array([0.2, 0.3, 0.5])
mu = np.array([-1.5, 0.75, 3.])
sigma = np.array([.75, .5, 1.])
comp = np.random.choice(mu.size, size=n_data, p=weight)
x_data = np.random.normal(mu[comp], sigma[comp], size=n_data)
plt.figure()
plt.hist(x_data, bins=200, label=r'Actual Data')

# inference
if __name__=='__main__':
    with pm.Model() as model:
        w=pm.Dirichlet('w', np.ones_like(weight))
        mu=pm.Normal('mu', 0., 10., shape=weight.size)
        tau=pm.Gamma('tau', 1., 1., shape=weight.size)
        x_observed = pm.NormalMixture('x_observed', w, mu, tau=tau, \
                                      observed=x_data)
        trace=pm.sample(5000, n_init=10000, tune=1000, random_seed=42)
    plt.figure()
    plt.hist(trace['w'], bins=50, label=r"posterior of $\weights$")
    plt.figure()
    plt.hist(trace['mu'], bins=50, label=r"posterior of $\mus$")
    plt.figure()
    plt.hist(trace['tau'], bins=50, label=r"posterior of $\taus$")
    with model:
        p_trace = pm.sample_posterior_predictive(trace, 5000, random_seed=42)
    plt.figure()
    plt.hist(p_trace['x_observed'], bins=50, density=True, histtype='step',\
        lw=2, alpha=.05, label='Posterior predictive distribution')
    plt.hist(x_data, bins=50, density=True, histtype='step', lw=2, \
        label='Observed data')
plt.show()