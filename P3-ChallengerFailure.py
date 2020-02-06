import pymc3 as pm 
import numpy as np
import os
import matplotlib.pyplot as plt 
import theano.tensor as tt 

path = os.path.join('datasets','challenger')
np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt(path+'\challenger_data.txt', skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]
plt.figure()
plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
            alpha=0.5)
# fitting sigmoid function
temperature = challenger_data[:,0]
D = challenger_data[:,1]
if __name__ == '__main__': # needs to be here bc some issues wit pm3(?)
    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0., tau=0.001, testval=0.)
        alpha = pm.Normal('alpha', mu=0., tau=0.001, testval=0.)
        p = pm.Deterministic('p', 1./(1.+tt.exp(beta*temperature+alpha)))
        observed = pm.Bernoulli('bernoulli_obs', p, observed=D)
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(120000, step = step, start=start)
        burned_trace = trace[100000::2] # 100k burnin, 2 to break the autocorrelation (if any)

    alpha_samples = burned_trace["alpha"][:, None]
    beta_samples = burned_trace["beta"][:, None]
    plt.figure()
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\beta$", color="#7A68A6", normed=True)
    plt.show()