import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataGen_DUCB import DataGen
from StoExp import StoExp
from DPFit import DPFit
from CausalBound import CausalBound

D = 5
N = 10000
Ns = 20
seed_num = np.random.randint(10000000)
Mode = 'crazy'

datagen = DataGen(D,N,Ns,Mode, seed_num)
datagen.data_gen()
Obs = datagen.Obs
Intv = datagen.Intv
#
# plt.figure('Intervention')
# Intv['Y'].plot(kind='density')
#
# plt.figure('Observation')
# Obs['Y'].plot(kind='density')

stoexp = StoExp(D)

# Obs
Z_obs, X_obs = stoexp.Construct_XZ(Obs)
obslogit = stoexp.Logit(X_obs,Z_obs)
obssvmlin = stoexp.SVM_linear(X_obs, Z_obs)
obssvmrbf = stoexp.SVM_rbf(X_obs, Z_obs)
obslda = stoexp.LDA(X_obs, Z_obs)
obsxgb = stoexp.XGB(X_obs, Z_obs)

# TrunOBS
TrunObs = stoexp.Trun_by_Y(Obs, 0.8, 0.5)
Z_tobs, X_tobs = stoexp.Construct_XZ(TrunObs)
tobslogit = stoexp.Logit(X_tobs,Z_tobs)
tobssvmlin = stoexp.SVM_linear(X_tobs, Z_tobs)
tobssvmrbf = stoexp.SVM_rbf(X_tobs, Z_tobs)
tobslda = stoexp.LDA(X_tobs, Z_tobs)
tobsxgb = stoexp.XGB(X_tobs, Z_tobs)

# TrunIntv
TrunIntv = stoexp.Trun_by_Y(Intv,0.8, 0.5)
Z_tintv, X_tintv = stoexp.Construct_XZ(TrunIntv)
tintvlogit = stoexp.Logit(X_tintv,Z_tintv)
tintvsvmlin = stoexp.SVM_linear(X_tintv, Z_tintv)
tintvsvmrbf = stoexp.SVM_rbf(X_tintv, Z_tintv)
tintvxgb = stoexp.XGB(X_tintv, Z_tintv)

# policy_list = [obslogit, obssvmlin, obssvmrbf, obslda,
#                tobslogit, tobssvmlin, tobssvmrbf, tobsxgb, tobslda,
#                tintvlogit, tintvsvmlin, tintvsvmrbf, tintvxgb
#                ]

policy_list = [obslogit,tobslogit,tintvxgb]

X_pl_list = []
Y_pl_list = []
for pl in policy_list:
    X_pl = np.round(pl.predict(Z_obs),0)
    Y_pl = datagen.gen_policy_Y(np.matrix(X_pl))
    X_pl_list.append(X_pl)
    Y_pl_list.append(Y_pl)
    print(np.mean(Y_pl))

C_list = []
for pl in policy_list:
    c_pl = stoexp.Compute_C(obsxgb,pl,Z_obs)
    C_list.append(c_pl)

Y_policy = pd.DataFrame(np.matrix(np.array(Y_pl_list).T))

# Initial true DPMM for obs
Yobs = Obs['Y']
init_compo = 50
DPFit_obs = DPFit(Yobs, init_compo)
DPFit_obs.Conduct()
dpobs = DPFit_obs.dpgmm
init_compo = sum(1 for x in dpobs.weights_ if x > 1e-2)
DPFit_obs = DPFit(Yobs, init_compo)
DPFit_obs.Conduct()
dpobs = DPFit_obs.dpgmm

iter_opt = 100
Bound_list = []
Model_list = []
Weight_list = []
True_Mu = []

for pl in range(len(policy_list)):
    Yintv_pl = Y_policy[pl]
    C_pl = C_list[pl]
    CB = CausalBound(dpobs, C_pl)
    LB, UB, lower, upper, lower_weight, upper_weight = CB.Solve_Optimization(C_pl, init_compo, iter_opt)
    Bound_list.append([LB,UB])
    Model_list.append([lower,upper])
    Weight_list.append([lower_weight, upper_weight])
    True_Mu.append(np.mean(Yintv_pl))

print(Bound_list)
print(True_Mu)

