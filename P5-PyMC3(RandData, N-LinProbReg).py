import numpy as np 
import pymc3 as pm 
import matplotlib.pyplot as plt 
import scipy.optimize as sop

seed = np.random.seed(42)
x = np.random.rand(100)*2.
x.sort()
nl = 1.5 + np.random.rand(1)
slope = 2. + np.random.rand(1)
intercept = np.random.rand(1)
rr =  np.random.rand(100)-0.5
y = nl * x**3 + slope * x + intercept + rr
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
    # inference
    alpha_samples = burned_trace["alpha"][:, None]
    beta_samples = burned_trace["beta"][:, None]
    plt.figure()
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.figure()
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\beta$", color="#7A68A6", normed=True)

    # define loss for a bayesian action inference
    def showdown_loss(guess, true_, risk, tol = 0.1):
        loss = np.zeros_like(true_)
        ix = true_ < guess
        loss[~ix] = np.abs(guess - true_[~ix])
        close_mask = [abs(true_ - guess) <= tol]
        loss[close_mask] = -2*true_[close_mask]
        loss[ix] = risk
        return loss
        
    guesses = np.linspace(0, 10, 100) 
    risks = np.logspace(0.001, 2.0, num=5)
    expected_loss = lambda guess, risk: showdown_loss(guess, burned_trace["alpha"], risk).mean()
    for risk in risks:
        results = [expected_loss(guess, risk) for guess in guesses]
        _min_results = sop.fmin(expected_loss, 1, args=(risk,), disp = False)
        plt.plot(guesses, results, label = "%d"%risk)
        plt.scatter(_min_results, 0, s = 60, label = "%d"%risk)
        plt.vlines(_min_results, 0, 100, linestyles="--")
        print("minimum at risk %d: %.2f" % (risk, _min_results))
    plt.title("Expected loss & Bayes actions of different guesses, \n \
    various risk-levels of overestimating")
    plt.legend(loc="upper left", scatterpoints = 1, title = "Bayes action at risk:")
    plt.xlabel("slope guess")
    plt.ylabel("expected loss")
    plt.xlim(0, 10)
    plt.ylim(-10, 110)
    plt.show()