from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np 

problem={'num_vars':2,\
    'names':['x','y'],\
        'bounds':[[-1,1], [0,2]]}
param_values = saltelli.sample(problem, 1000)
z = np.zeros([param_values.shape[0]])

def evaluate_model(P):
    return 2.*P[0]+P[1]**2+np.exp(P[0]*P[1])

for i, P in enumerate(param_values):
    z[i] = evaluate_model(P)

Si = sobol.analyze(problem, z)
print('First-Order Sensitivity:\n{}\
    \nSecond-Order Sensitivity:\n{}\
    \nTotal-Order Sensitivity:\n{}'\
    .format(Si['S1'],Si['S2'],Si['ST']))

