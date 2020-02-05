import pymc3 as pm
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p_true = 0.05
    N = 1000
    occurrences = stats.bernoulli.rvs(p_true, size=N)
    with pm.Model() as model:
        p = pm.Uniform('p', lower=0, upper=1)
        obs = pm.Bernoulli("obs", p, observed=occurrences)
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        burned_trace = trace[1000:]
    plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    plt.hist(burned_trace["p"], bins=25, histtype="stepfilled", normed=True)
    plt.show()