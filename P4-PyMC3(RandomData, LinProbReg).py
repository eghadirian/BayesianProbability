import numpy as np 
import pymc3 as pm 
import matplotlib.pyplot as plt 

seed = np.random.seed(42)
x = np.random.rand(100)
x.sort()
slope = 2. + np.random.rand(1)
intercept = np.random.rand(1)
rr =  np.random.rand(100)-0.5
y = slope * x + intercept + rr
plt.figure()
plt.plot(x, y)

def linear(x, slope, inter=0.):
    return (slope * x) + inter

if __name__=='__main__':
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0., sigma=1., testval=0.)
        beta = pm.Normal('beta', mu=0., sigma=1., testval=0.)
        sigma = pm.HalfCauchy('sigma', beta=10., testval=1.)
        y_ = pm.Deterministic('y', linear(x, alpha, beta))
        observed = pm.Normal('bernoulli_obs', mu=y_, sigma=sigma, observed=y)
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(120000, step=step, start=start)
        burned_trace = trace[100000::2]

    alpha_samples = burned_trace["alpha"][:, None]
    beta_samples = burned_trace["beta"][:, None]
    plt.figure()
    plt.hist(alpha_samples, histtype='stepfilled',
            label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(beta_samples, histtype='stepfilled',
            label=r"posterior of $\beta$", color="#7A68A6", normed=True)
    plt.show()
    print('slope: {}, intercept: {}'.format(slope, intercept))