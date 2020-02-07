import pymc3 as pm 
import numpy as np
import os
import matplotlib.pyplot as plt 
import theano.tensor as tt 
from scipy.stats.mstats import mquantiles

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

def logistic(x, beta, alpha=0):
    return 1./(1. + np.exp(np.dot(beta, x) + alpha))

if __name__ == '__main__': # needs to be here bc some issues wit pm3(?)
    with pm.Model() as model:
        beta = pm.Normal('beta', mu=0., tau=0.001, testval=0.)
        alpha = pm.Normal('alpha', mu=0., tau=0.001, testval=0.)
        p = pm.Deterministic('p', 1./(1.+tt.exp(beta*temperature+alpha)))
        observed = pm.Bernoulli('bernoulli_obs', p, observed=D)
        # artificially generate credible data sets, from POSTERIOR (not the prior)
        simulated_data = pm.Bernoulli('simulated-data', p, shape=p.tag.test_value.shape)
        start = pm.find_MAP()
        step = pm.Metropolis(vars=[p])
        trace = pm.sample(120000, step = step, start=start)
        burned_trace = trace[100000::2] # 100k burnin, 2 to break the autocorrelation (if any)

    alpha_samples = burned_trace["alpha"][:, None]
    beta_samples = burned_trace["beta"][:, None]
    simulations = burned_trace["bernoulli_sim"]
    plt.figure()
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\beta$", color="#7A68A6", normed=True)

    t = np.linspace(temperature.min() - 5, temperature.max()+5, 50)[:, None]
    p_t = logistic(t.transpose(), beta_samples, alpha_samples)
    mean_prob_t = p_t.mean(axis=0)
    # 95% confidence interval
    qs = mquantiles(p_t, [0.025, 0.975], axis=0)
    plt.figure()
    plt.fill_between(t[:, 0], *qs, alpha=0.7,
                    color="#7A68A6")
    plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)
    plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
            label="average posterior \nprobability of defect")
    temp = input('At what temperature (degF) you will be launching?')
    prob_ = logistic(float(temp), beta_samples, alpha_samples)
    plt.xlim(0., 1.)
    plt.ylim()
    plt.hist(prob_, bins=1000, normed=True, histtype='stepfilled')
    plt.title("Posterior distribution of probability of defect, given $t = {}$".format(temp))
    plt.xlabel("probability of defect occurring in O-ring")
    plt.figure()
    plt.scatter(temperature, simulations[1000, :], color="k",
                s=50, alpha=0.6)
    plt.show()

    
    