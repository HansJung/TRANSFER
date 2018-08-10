import numpy as np
import pandas as pd
from UCB import UCB
from KLUCB import KLUCB
import Sim_GenData
import matplotlib.pyplot as plt

X = 'RXASP'
K = 2
T = 4500

EXP,OBS = Sim_GenData.RunGenData()
LB,HB = Sim_GenData.ComputeBound(OBS,X)
lx0,lx1 = LB
hx0,hx1 = HB
bound_list = [[lx0,hx0],[lx1,hx1]]

EXP = Sim_GenData.ChangeRXASPtoX(EXP)
OBS = Sim_GenData.ChangeRXASPtoX(OBS)
ucb = UCB(bound_list,EXP,K,T)
klucb = KLUCB(bound_list,EXP,K,T)

# [[prob_opt, cum_regret, UCB_list, Arm,X_hat_list, Na_T],[prob_opt_B, cum_regret_B, UCB_list_B, UCB_hat_list_B, Arm_B, X_hat_list_B, Na_T_B]] = ucb.Bandit_Run()
[[prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T],[prob_opt_list_B, cum_regret_list_B, UCB_list_B, Arm_B, X_hat_list_B, Na_T_B]] = klucb.Bandit_Run()

plt.figure(1)
plt.title('Prob Opt')
plt.plot(prob_opt_list,label='KLUCB')
plt.plot(prob_opt_list_B, label='B-KLUCB')
plt.legend()

plt.figure(2)
plt.title('Cummul. Regret')
plt.plot(cum_regret_list,label='KLUCB')
plt.plot(cum_regret_list_B, label='B-KLUCB')
plt.legend()

plt.show()