from Experiment.DataGen import DataGen
from Experiment.DPFit import DPFit
from Experiment.CausalBound import CausalBound
from Experiment.UCB import UCB
import numpy as np
import matplotlib.pyplot as plt

''' From DataGen '''
D = 100
N = 20000
Ns = 20
Mode = 'crazy'

datagen = DataGen(D,N,Ns,Mode)
datagen.data_gen()
Obs = datagen.Obs
Intv = datagen.Intv

plt.figure('Intervention')
Intv[Intv['X']==0]['Y'].plot(kind='density')
Intv[Intv['X']==1]['Y'].plot(kind='density')

plt.figure('Observation')
Obs[Obs['X']==0]['Y'].plot(kind='density')
Obs[Obs['X']==1]['Y'].plot(kind='density')


# plt.figure()

''' From DPFit '''
init_compo = 10
bound_list = []

for x in [0,1]:
    Yobs_x = Obs[Obs['X']==x]['Y']
    Yintv_x = Intv[Intv['X']==x]['Y']

    DPFit_obs = DPFit(Yobs_x,init_compo)
    DPFit_intv = DPFit(Yintv_x,init_compo)
    DPFit_obs.Conduct()
    dpobs = DPFit_obs.dpgmm
    DPFit_intv.Conduct()
    dpintv = DPFit_intv.dpgmm

    ''' From Causal Bound '''
    px = len(Obs[Obs['X'] == x])/ N
    C =  -np.log(px)
    CB = CausalBound(dpobs,C)
    opt_result = CB.Solve_Optimization(C,init_compo)
    bound_list.append(opt_result)

print(bound_list)
print(np.mean(Intv[Intv['X']==0]['Y']), np.mean(Intv[Intv['X']==1]['Y']))

''' From UCB '''
K = 1
ucb = UCB(bound_list,Intv,K)
prob_opt_list, cum_regret_list, prob_opt_list_B, cum_regret_list_B = ucb.Bandit_Run()

print(prob_opt_list)
print(prob_opt_list_B)

print(cum_regret_list)
print(cum_regret_list_B)

plt.figure()
plt.title('Cum regret')
plt.plot(cum_regret_list,label='UCB')
plt.plot(cum_regret_list_B,label='B-UCB')
plt.legend()

plt.figure()
plt.title('Prob_opt_list')
plt.plot(prob_opt_list,label='UCB')
plt.plot(prob_opt_list_B,label='B-UCB')
plt.legend()
