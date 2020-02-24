import pymc3 as pm 
import numpy as np 
import theano.tensor as tt
import matplotlib.pyplot as plt 

# Mock data generator
np.random.seed(seed=42)
rand = np.random.normal(loc=0., scale=.1, size=10)
x_1 = np.arange(10)
x_2 = np.sqrt(x_1)
y_1 = x_1 + (1. + rand) # alpha_1 * x_1 + beta_1
y_2 = (1. + rand)*(x_1 * x_2) # alpha_2 * x_1 * x_2
y_3 = np.sin(x_2 ** (1. + rand)) # alpha_3 * sin(x_2 ^ gamma_1)

# Regression functions
def encapsulate(x_1, x_2, alpha_1, alpha_2, alpha_3, beta_1, gamma_1):
    def fun_lin(x_1, alpha_1, beta_1): 
      return alpha_1 * x_1 + beta_1
    def fun_nlin_bi(x_1, x_2, alpha_2): 
        return alpha_2 * x_1 * x_2
    def fun_nlin_un(x_2, alpha_3, gamma_1): 
        return alpha_3 * np.sin(x_2 ** gamma_1)
    return (fun_lin(x_1, alpha_1, beta_1), \
            fun_nlin_bi(x_1, x_2, alpha_2), \
            fun_nlin_un(x_2, alpha_3, gamma_1))

# Bayesian inference
if __name__=='__main__':
    with pm.Model() as model:
        alpha_1 = pm.Normal('alpha1', mu=0., sigma=1., testval=0.)
        alpha_2 = pm.Normal('alpha2', mu=0., sigma=1., testval=0.)
        alpha_3 = pm.Normal('alpha3', mu=0., sigma=1., testval=0.)
        beta_1 = pm.Normal('beta1', mu=0., sigma=1., testval=0.)
        gamma_1 = pm.Gamma('gamma1', alpha=5., beta=5.)
        y1_det, y2_det, y3_det = \
                encapsulate(x_1, x_2, alpha_1, alpha_2, alpha_3, beta_1, gamma_1)
        y1 = pm.Deterministic('y1', y1_det)
        y2 = pm.Deterministic('y2', y2_det)
        y3 = pm.Deterministic('y3', y3_det)
        observed = pm.Normal('obs', tt.stack([y1, y2, y3]),\
            observed=tt.stack([y_1, y_2, y_3]))
        trace = pm.sample(120000, step=pm.NUTS(), start=pm.find_MAP())
        burned_trace = trace[100000::2]
    alpha1_samples = burned_trace["alpha1"][:, None]
    alpha2_samples = burned_trace["alpha2"][:, None]
    alpha3_samples = burned_trace["alpha3"][:, None]
    beta1_samples = burned_trace["beta1"][:, None]
    gamma1_samples = burned_trace["gamma1"][:, None]
    plt.subplot(3,2,1)
    plt.hist(alpha1_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha1$", color="#A60628", normed=True)
    plt.subplot(3,2,3)
    plt.hist(alpha2_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha2$", color="#A60628", normed=True)
    plt.subplot(3,2,5)
    plt.hist(alpha3_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\alpha3$", color="#A60628", normed=True)
    plt.subplot(3,2,2)
    plt.hist(beta1_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\beta1$", color="#A60628", normed=True)
    plt.subplot(3,2,6)
    plt.hist(gamma1_samples, histtype='stepfilled', bins=35, alpha=0.85,
            label=r"posterior of $\gamma1$", color="#A60628", normed=True)
    plt.show()