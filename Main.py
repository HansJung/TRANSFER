from DataGen import DataGen
from DPFit import DPFit
from CausalBound import CausalBound
from UCB import UCB
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
init_compo = 25
iter_opt = 1000
bound_list = []
bounded_models = []

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
    LB,UB,lower,upper = CB.Solve_Optimization(C,init_compo, iter_opt)
    bound_list.append([LB,UB])
    bounded_models.append([lower,upper])

print('')
print('Observational mean')
print('X=0',np.mean(Obs[Obs['X']==0]['Y']))
print('X=1',np.mean(Obs[Obs['X']==1]['Y']))
print('')

print(bound_list)
print(np.mean(Intv[Intv['X']==0]['Y']), np.mean(Intv[Intv['X']==1]['Y']))

''' From UCB '''
K = 1
ucb = UCB(bound_list,Intv,K)
[UCB_result, BUCB_result] = ucb.Bandit_Run()
prob_opt, cum_regret, UCB_list, Arm, X_hat, Na_T = UCB_result
prob_opt_B, cum_regret_B, UCB_list_B, UCB_hat_list_B, Arm_B, X_hat_B, Na_T_B = BUCB_result

print(prob_opt)
print(prob_opt_B)

print(cum_regret)
print(cum_regret_B)

plt.figure()
plt.title('Cum regret')
plt.plot(cum_regret,label='UCB')
plt.plot(cum_regret_B,label='B-UCB')
plt.legend()

plt.figure()
plt.title('Prob_opt_list')
plt.plot(prob_opt,label='UCB')
plt.plot(prob_opt_B,label='B-UCB')
plt.legend()
