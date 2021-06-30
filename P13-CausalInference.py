from causalimpact import CausalImpact 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

t = np.arange(30)
X = np.random.normal(0, 0.1, 30)
Y = np.random.normal(5, 1, 30)
data = pd.DataFrame({'X':X, 'Y':Y})[['Y', 'X']]
data.iloc[25:30, 0] += np.arange(5, 0, -1) #do an impact
pre_period = [0, 24]
post_period = [25, 29]

ci = CausalImpact(data=data, pre_period=pre_period, post_period=post_period)

