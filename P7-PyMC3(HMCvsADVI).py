import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import warnings
from numpy.random import binomial, randn, uniform
from sklearn.model_selection import train_test_split
import seaborn as sns

if __name__=='__main__':
    sns.set()
    warnings.filterwarnings('ignore')
 # generating mock data
    alpha_0 = 1.
    alpha_1 = 1.25
    beta_0 = 1.
    beta_1 = 1.25
    gamma = 0.75
    n_samples = 1000
    category = binomial(n=1, p=0.5, size=n_samples)
    x = uniform(low=0, high=10, size=n_samples)
    y = ((1 - category) * alpha_0 + category * alpha_1\
        + ((1 - category) * beta_0 + category * beta_1) * x\
            + gamma * randn(n_samples))
    model_data = pd.DataFrame({'y': y, 'x': x, 'category': category})

    # split the data
    train, test = train_test_split(\
        model_data, test_size=0.2, stratify=model_data.category)
    y_tensor = theano.shared(train.y.values.astype('float64'))
    x_tensor = theano.shared(train.x.values.astype('float64'))
    cat_tensor = theano.shared(train.category.values.astype('int64'))

    # HMC inference
    with pm.Model() as model:
        alpha_prior = pm.HalfNormal('α', sd=2, shape=2)
        beta_prior = pm.Normal('β', mu=0, sd=2, shape=2)
        gamma_prior = pm.HalfNormal('𝞂', sd=2, shape=1)
        mu_likelihood = alpha_prior[cat_tensor] + beta_prior[cat_tensor] * x_tensor
        y_likelihood = pm.Normal('y', mu=mu_likelihood, sd=gamma_prior, observed=y_tensor)
        hmc_trace = pm.sample(draws=5000, tune=1000, cores=2) # NUTS HMC

        # ADVI
        map_tensor_batch = {y_tensor: pm.Minibatch(train.y.values, 100),\
                            x_tensor: pm.Minibatch(train.x.values, 100),\
                            cat_tensor: pm.Minibatch(train.category.values, 100)} # define mini-batches
        advi_fit = pm.fit(method='advi', n=30000,\
            more_replacements=map_tensor_batch) # VI inference: learn the relationship, similar to ML
        advi_trace = advi_fit.sample(10000)

        # testing each algorithm
        x_tensor.set_value(test.x.values)
        cat_tensor.set_value(test.category.values.astype('int64'))
        hmc_posterior_pred = pm.sample_ppc(hmc_trace, 1000, model)
        hmc_predictions = np.mean(hmc_posterior_pred['y'], axis=0)
        advi_posterior_pred = pm.sample_ppc(advi_trace, 1000, model)
        advi_predictions = np.mean(advi_posterior_pred['y'], axis=0)
        prediction_data = pd.DataFrame(\
            {'HMC': hmc_predictions, \
            'ADVI': advi_predictions, \
            'actual': test.y,\
            'error_HMC': hmc_predictions - test.y, \
            'error_ADVI': advi_predictions - test.y})
        _ = sns.lmplot(y='ADVI', x='HMC', data=prediction_data,
                    line_kws={'color': 'red', 'alpha': 0.5})

        # look into inference
        param_samples_HMC = pd.DataFrame(\
            {'α_0': hmc_trace.get_values('α')[:, 0], \
            'β_0': hmc_trace.get_values('β')[:, 0]})
        _ = sns.scatterplot(x='α_0', y='β_0', data=param_samples_HMC).set_title('HMC')
        param_samples_ADVI = pd.DataFrame(\
            {'α_0': advi_trace.get_values('α')[:, 0], \
            'β_0': advi_trace.get_values('β')[:, 0]})
        _ = sns.scatterplot(x='α_0', y='β_0', data=param_samples_ADVI).set_title('ADVI')
        # see the impact of ADVI's assumption of n-dimensional spherical Gaussians.

        # compare with actual data
        RMSE = np.sqrt(np.mean(prediction_data.error_ADVI ** 2))
        print(f'RMSE for ADVI predictions = {RMSE:.3f}')
        _ = sns.lmplot(y='ADVI', x='actual', data=prediction_data, \
                    line_kws={'color': 'red', 'alpha': 0.5})
