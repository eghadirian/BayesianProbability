import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import scipy as sp 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import seaborn as sns
import pandas as pd
import warnings
import pymc3.distributions.transforms as tr
import numpy as np

class SampleGenerator:
    """ Generate simulated data """
    PROCESSES = ['poisson']

    def __init__(self, process, params, n_state, transition_mat):
        if (len(params) != n_state):
            raise ValueError("params count: {} is not equal to n_state: {}".format(len(params), n_state))
        if (transition_mat.shape[0] != n_state) or (transition_mat.shape[0] != transition_mat.shape[1]):
            raise ValueError("`transition_mat` is not square or not equal to `n_state`")
        if not np.isclose(transition_mat.sum(axis=1), 1).all():
            raise ValueError("`transition_mat` rows should add to 1")
        if process not in self.PROCESSES:
            raise NotImplementedError("`process` type of {} is not implemented".format(process))
        self.process_type = process
        self.process_params = params
        self.n_state = n_state
        self.transition_mat = transition_mat

    def __validate_inputs(self, n_samples, init_state):
        if init_state >= self.n_state:
            raise ValueError("`init_state` is greater than `n_state`:{}".format(init_state))

    def __getsample(self, params):
        if self.process_type == 'poisson':
            sample = np.random.poisson(params['lambda'])
        else:
            raise NotImplementedError("Process type not implemented")
        return sample

    def generate_samples(self, n_samples, seed = 42, init_state = 0, transition_distribution = 'uniform'):
        self.__validate_inputs(n_samples, init_state)
        curr_state = init_state
        state_history = []
        all_samples = []
        for sample_id in range(n_samples): 
            all_samples.append(self.__getsample(self.process_params[curr_state]))
            state_history.append(curr_state)
            transition_probs = self.transition_mat[curr_state]
            draw = np.random.uniform()
            highs = transition_probs.cumsum()
            lows = np.roll(highs, shift=1)
            lows[0] = 0
            for i, (low, high) in enumerate(zip(lows, highs)):
                if (draw >= low) and (draw < high):
                    curr_state = i
                    break
        return np.array(all_samples), np.array(state_history)

# Problem 1
sg = SampleGenerator("poisson", [{'lambda':5}, {'lambda': 10}], 2, 
                     np.array([[0.8, 0.2],[0.2, 0.8]]))
vals_simple, states_orig_simple = sg.generate_samples(100)

class StateTransitions(pm.Categorical):
    '''
    Distribution of state
    '''
    def __init__(self, trans_prob=None, init_prob=None, * args, ** kwargs):
        super(pm.Categorical, self).__init__(* args, ** kwargs)
        self.trans_prob = trans_prob
        self.init_prob = init_prob
        # Housekeeping
        self.mode = tt.cast(0,dtype='int64')
        self.k = 2
        
    def logp(self, x):
        trans_prob = self.trans_prob
        p = trans_prob[x[:-1]] # probability of transitioning based on previous state
        x_i = x[1:]            # the state you end up in
        log_p = pm.Categorical.dist(p, shape=(self.shape[0],2)).logp_sum(x_i)
        return pm.Categorical.dist(self.init_prob).logp(x[0]) + log_p

class PoissionProcess(pm.Discrete):
    def __init__(self, state=None, lambdas=None, *args, **kwargs):
        super(PoissionProcess, self).__init__(*args, **kwargs)
        self.state = state
        self.lambdas = lambdas
        # Housekeeping
        self.mode = tt.cast(1,dtype='int64')
    
    def logp(self, x):
        lambd = self.lambdas[self.state]
        llike = pm.Poisson.dist(lambd).logp_sum(x)
        return llike

chain_tran = tr.Chain([tr.ordered])
with pm.Model() as m:
    lambdas = pm.Gamma('lam0', mu = 10, sd = 100, shape = 2, transform=chain_tran,  testval=np.asarray([1., 1.5]))
    init_probs = pm.Dirichlet('init_probs', a = tt.ones(2), shape=2)
    state_trans = pm.Dirichlet('state_trans', a = tt.ones(2), shape=(2,2))
    states = StateTransitions('states', state_trans, init_probs, shape=len(vals_simple))
    y = PoissionProcess('Output', states, lambdas, observed=vals_simple)
    trace = pm.sample(tune=2000, sample=1000, chains=2)

lamb = trace["lamb0"][:, None]
plt.figure()
plt.hist(lamb, histtype='stepfilled', bins=35, alpha=0.85,
        label=r"posterior", color="#A60628", normed=True)
plt.show()