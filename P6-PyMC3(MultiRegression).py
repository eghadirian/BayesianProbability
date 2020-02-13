import numpy as np 
import pymc3 as pm 
import matplotlib.pyplot as plt 
import scipy.optimize as sop
import theano.tensor as tt
# Mock data generator
x = np.random.rand(10)
x.sort()
y1 = [1, 3, 1, 2, 8, 4, 6, 11, 9, 11]
y2 = [1, 3, 1, 2, 8, 4, 6, 11, 9, 11]**np.sqrt(2)

# Regression function
def linear(x, slope, inter=0., p = 1):
    return ((slope * x) + inter)**np.sqrt(p)

# Bayesian inference
if __name__=='__main__':
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0., sigma=1., testval=0.)
        beta = pm.Normal('beta', mu=0., sigma=1., testval=0.)
        sigma = pm.HalfCauchy('sigma', beta=10., testval=1.)
        param1 = pm.HalfNormal('param', sd=5.)
        param2 = pm.HalfNormal('param2', sd=5.)
        y_1 = pm.Deterministic('y', linear(x, alpha, beta, param1))
        y_2 = pm.Deterministic('y2', linear(x, alpha, beta, param2))
        observed = pm.Normal('bernoulli_obs', tt.stack([y_1, y_2]),\
                             sigma=sigma, observed=tt.stack([y1, y2]))
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(10000, step=step, start=start)
        step = pm.NUTS()
        trace = pm.sample(110000, step=step)
        burned_trace = trace[100000::2]
    alpha_samples = burned_trace["alpha"][:, None]
    param1_samples = burned_trace["param"][:, None]
    param2_samples = burned_trace["param2"][:, None]
    plt.figure()
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(param1_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\param1$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(param2_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\param2$", color="#A60628", normed=True)