import pandas as pd
import numpy as np
from scipy.optimize import minimize


def BinoKL(mu_hat, mu):
    return mu_hat * np.log(mu_hat/mu) + (1-mu_hat) * np.log((1-mu_hat)/(1-mu))

def ConstFun(mu_hat,mu,ub):
    ConstVal = -1 * (BinoKL(mu_hat, mu) - ub)
    if ConstVal > 0:
        return True
    else:
        return False

mu_hat = 0.2
ObjFun = lambda mu: -1*BinoKL(mu_hat, mu)
ObjFunDer = lambda mu: mu_hat/mu - (1-mu_hat)*(1/(1-mu))
ObjFunHess = lambda mu: -mu_hat/(mu**2) - (1-mu_hat)/((1-mu)**2)

ub = 0.9
constraints = {'type':'ineq',
               'fun': lambda mu: -1*(BinoKL(mu_hat, mu) - ub),
               'jac': lambda mu: mu_hat/mu - (1-mu_hat)*(1/(1-mu))
               }

while 1:
    x0 = np.random.uniform(low=mu_hat,high=1,size=1)
    if ConstFun(mu_hat,x0,ub) == True:
        break


bound = tuple([tuple([0,1])])
res = minimize(ObjFun, [x0], method='SLSQP', jac=ObjFunDer, hess=ObjFunHess,
               constraints=constraints, bounds = bound, options={'ftol':1e-8,'gtol':1e-8, 'disp':True})
