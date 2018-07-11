import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from DataGen import DataGen
from StoExp import StoExp
from JZ_bound import JZ_bound
from DUCB import DUCB

# Parameter configuration
D = 1
N = 5000
T = int(N/2)
# seed_num = np.random.randint(10000000)
seed_num = 7308595 # Case 2
# seed_num = 3478042 # Case 3

# D=1
## Case 3
### seed_num = 3478042 // D=1, N=5000, T=N/2

## Case 2
### seed_num = 1205748 // D=1, N=5000, T=N/2
### seed_num = 7308595 // D=1, N=5000, T=N/2



# Generating Observation data
datagen = DataGen(D,N,seed_num)
OBS = datagen.obs_data()

# Externally policy
stoexp = StoExp(D)
JZ = JZ_bound()
[X,Y,Z] = JZ.sepOBS(OBS,D)

obslogit = stoexp.Logit(1-X,Z)
obsxgb = stoexp.XGB(X,Z)

policy_list = [obslogit,obsxgb]

## Policy data
poly_obslogit_data = datagen.poly_intv_data(obslogit, Z)
poly_obsxgb_data = datagen.poly_intv_data(obsxgb, Z)
X_pl_list = [poly_obslogit_data['X'], poly_obsxgb_data['X']]
Y_pl_list = [poly_obslogit_data['Y'], poly_obsxgb_data['Y']]

obslogit_pred = obslogit.predict_proba(Z)
obsxgb_pred = obsxgb.predict_proba(Z)
pred_list = [obslogit_pred, obsxgb_pred]

### True mean
Y_obslogit = np.mean(poly_obslogit_data['Y'])
Y_obsxgb = np.mean(poly_obsxgb_data['Y'])
Y_pis = [Y_obslogit, Y_obsxgb]

opt_pl = np.argmax([Y_obslogit, Y_obsxgb])
subopt_pl = 1-opt_pl
opt_Ypi = Y_pis[opt_pl]
subopt_Ypi = Y_pis[subopt_pl]

# Bound construction
JZ = JZ_bound()
[L_obslogit, H_obslogit] = JZ.JZ_bounds(obslogit,OBS,D,N)
[L_obsxgb, H_obsxgb] = JZ.JZ_bounds(obsxgb,OBS,D,N)
Bdd = [[L_obslogit, H_obslogit],[L_obsxgb, H_obsxgb]]

hx = Bdd[subopt_pl][1]
if hx < opt_Ypi:
    case_num = 2
else:
    case_num = 3

# DUCB
ducb = DUCB(policy_list,pred_list,opt_pl,T, X_pl_list,Y_pl_list,Z)
prob_opt_list,avg_loss_list,num_pull = ducb.conduct_DUCB()
bdd_prob_opt_list,bdd_avg_loss_list,bdd_num_pull = ducb.conduct_BDUCB(Bdd)

print(case_num)

plt.figure()
plt.title("Prob optimal policy")
plt.plot(prob_opt_list,label="DUCB")
plt.plot(bdd_prob_opt_list,label="B-DUCB")
plt.legend()

plt.figure()
plt.title("Average loss")
plt.plot(avg_loss_list,label="DUCB")
plt.plot(bdd_avg_loss_list,label="B-DUCB")
plt.legend()


