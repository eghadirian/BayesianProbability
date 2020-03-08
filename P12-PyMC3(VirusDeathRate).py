import pandas as pd 
import matplotlib.pyplot as plt 
import pymc3 as pm 
from google.colab import files

# from https://www.worldometers.info/coronavirus/
uploaded = files.upload()
df = pd.read_excel('Book1.xlsx').fillna(0)
number = df['Total Cases']
percent = df['Total Deaths']/df['Total Cases']

occurances = []
for i in range(len(number)):
    for j in range(number[i]):
        occurances.append(percent[i])

if __name__=='__main__':
    with pm.Model() as modl:
        p = pm.Normal('Death Rate', mu=0.02, sigma=0.005)
        sigma = pm.Gamma('sigma', 1., 1.)
        obs = pm.Normal('observation', mu=p, sigma=sigma, observed=occurances)
        trace = pm.sample(200000, step=pm.Metropolis())
        burned_trace = trace[150000::2]
    mu_samples = burned_trace['Death Rate'][:, None]
    s_samples = burned_trace['sigma'][:, None]
    plt.figure()
    plt.hist(mu_samples, bins=30, label=r"posterior of $\Death Rate$", \
             color="#A60628", normed=True)
    plt.figure()
    plt.hist(s_samples, bins=30, label=r"posterior of $\Standard Deviation$", \
             color="#A60628", normed=True)



