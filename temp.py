import numpy as np
import copy

def BinoKL(mu_hat, mu):
    return mu_hat * np.log(mu_hat / mu) + (1 - mu_hat) * np.log((1 - mu_hat) / (1 - mu))

def BinarySearchMax(mu_hat, M, init_maxval=1):
    maxval = copy.copy(init_maxval)
    mu = mu_hat
    terminal_cond = 1e-8
    while 1:
        mu_cand = (mu + maxval) / 2
        KL_val = BinoKL(mu_hat, mu_cand)
        diff = np.abs(KL_val - M)
        if KL_val < M:
            if diff < terminal_cond:
                mu = mu_cand
                return mu
            else:
                mu = copy.copy(mu_cand)
        else:
            maxval = copy.copy(mu_cand)
        if np.abs(mu-1) < terminal_cond:
            return mu

mu_hat = 0.1809
M = 14.7276071649
mu = BinarySearchMax(mu_hat=mu_hat,M=M,init_maxval=1)

print(mu)
print(BinoKL(mu_hat,mu))


#
# M = 0.8
# mu_hat = 0.2
#
# maxval = 1
# mu = mu_hat
# minval = mu_hat
#
# iteridx = 0
# while 1:
#     iteridx += 1
#     mu_cand = (mu + maxval)/2
#     KL_val = BinoKL(mu_hat,mu_cand)
#     diff = np.abs(KL_val - M)
#     if diff > 1e-10:
#         if KL_val < M: # Go to right
#             minval = copy.copy(mu_cand)
#             mu = copy.copy(mu_cand)
#         elif KL_val > M: # Go to left
#             maxval = copy.copy(mu_cand)
#     else:
#         break
#     print(mu,mu_cand,minval,maxval)
#     if iteridx > 30:
#         break


















# import numpy as np
# import scipy as sp
#
# t = 20
# num_policy = 5
#
# # Given Time index
# timeidx = list(range(1,t+1))
#
# # Given numPl
# numPl = {1:4,2:3,3:5,4:0,5:8}
#
#
# dictlistPl = {1:[3,5,7,10],2:[1,6,12],3:[2,8,11,17,20],4:[],5:[4,9,13,14,15,16,18,19]}
# propPl = {1:4/t,2:3/t,3:5/t,4:0/t,5:8/t}
#
# num_group = int(np.floor(4*np.log(t)))
# perm_timeidx = np.array_split(np.random.permutation(timeidx),num_group)
#
# listGroupPl = list()
# for r in range(num_group):
#     dict_eachgroupPl = dict()
#     listGroupTime = perm_timeidx[r]
#     for pl in range(1,num_policy+1):
#         listTimePl = dictlistPl[pl]
#         dict_eachgroupPl[pl] = list(set(listTimePl) & set(listGroupTime))
#     listGroupPl.append(dict_eachgroupPl)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import pandas as pd
# # import numpy as np
# # from scipy.optimize import minimize
# #
# #
# # def BinoKL(mu_hat, mu):
# #     return mu_hat * np.log(mu_hat/mu) + (1-mu_hat) * np.log((1-mu_hat)/(1-mu))
# #
# # def ConstFun(mu_hat,mu,ub):
# #     ConstVal = -1 * (BinoKL(mu_hat, mu) - ub)
# #     if ConstVal > 0:
# #         return True
# #     else:
# #         return False
# #
# # mu_hat = 0.2
# # ObjFun = lambda mu: -1*BinoKL(mu_hat, mu)
# # ObjFunDer = lambda mu: mu_hat/mu - (1-mu_hat)*(1/(1-mu))
# # ObjFunHess = lambda mu: -mu_hat/(mu**2) - (1-mu_hat)/((1-mu)**2)
# #
# # ub = 0.9
# # constraints = {'type':'ineq',
# #                'fun': lambda mu: -1*(BinoKL(mu_hat, mu) - ub),
# #                'jac': lambda mu: mu_hat/mu - (1-mu_hat)*(1/(1-mu))
# #                }
# #
# # while 1:
# #     x0 = np.random.uniform(low=mu_hat,high=1,size=1)
# #     if ConstFun(mu_hat,x0,ub) == True:
# #         break
# #
# #
# # bound = tuple([tuple([0,1])])
# # res = minimize(ObjFun, [x0], method='SLSQP', jac=ObjFunDer, hess=ObjFunHess,
# #                constraints=constraints, bounds = bound, options={'ftol':1e-8,'gtol':1e-8, 'disp':True})
