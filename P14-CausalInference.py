from causalinference import CausalModel
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

num = 1000
effect = 100
treatment = 100
X = np.random.normal(0, 0.1, num)
Y = np.random.normal(effect, 1, num)
Y[num-effect:] += treatment
D = np.concatenate((np.zeros(num-effect+1), np.ones(effect-1)))

cm = CausalModel(Y, D, X)
print(cm.summary_stats)

cm.est_via_ols(1)
print(cm.estimates)




