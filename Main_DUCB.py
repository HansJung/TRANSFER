import pandas as pd
import numpy as np
from DUCB import DUCB
import GenData_IST
import matplotlib.pyplot as plt

X = 'RXASP'
K = 2
T = 5500

EXP,OBS = GenData_IST.RunGenData()
# print(GenData_IST.QualityCheck(EXP,OBS,X))
print(GenData_IST.ObsEffect(EXP,'Y'))
print(GenData_IST.ObsEffect(OBS,'Y'))

LB,HB = GenData_IST.EmpiricalComputeBound(OBS,X,delta=0.01)
lx0,lx1 = LB
hx0,hx1 = HB
bound_list = [[lx0,hx0],[lx1,hx1]]

EXP = GenData_IST.ChangeRXASPtoX(EXP,idx_X=2)
OBS = GenData_IST.ChangeRXASPtoX(OBS,idx_X=2)

policy_list = []

prob_zero = 0.01
prob_one = 0.99
pl1 = lambda age, sex: prob_one if (age == 0) and (sex == 0) else prob_zero
pl2 = lambda age, sex: prob_one if (age == 0) and (sex == 1) else prob_zero
pl3 = lambda age, sex: prob_one if (age == 1) and (sex == 0) else prob_zero
pl4 = lambda age, sex: prob_one if (age == 1) and (sex == 1) else prob_zero

policy_list = [pl1,pl2,pl3,pl4]

# pl1 = lambda age, sex: 1 if (age == 0) and (sex == 0) else 0
# pl2 = lambda age, sex: 1 if (age == 0) and (sex == 1) else 0
# pl3 = lambda age, sex: 1 if (age == 1) and (sex == 0) else 0
# pl4 = lambda age, sex: 1 if (age == 1) and (sex == 1) else 0
# pl12 = lambda age, sex: 1 if ((age == 0) and (sex == 0)) or ((age == 0) and (sex == 1)) else 0
# pl13 = lambda age, sex: 1 if ((age == 0) and (sex == 0)) or ((age == 1) and (sex == 0)) else 0
# pl14 = lambda age, sex: 1 if ((age == 0) and (sex == 0)) or ((age == 1) and (sex == 1)) else 0
# pl23 = lambda age, sex: 1 if ((age == 0) and (sex == 1)) or ((age == 1) and (sex == 0)) else 0
# pl24 = lambda age, sex: 1 if ((age == 0) and (sex == 1)) or ((age == 1) and (sex == 1)) else 0
# pl34 = lambda age, sex: 1 if ((age == 1) and (sex == 0)) or ((age == 1) and (sex == 1)) else 0



X_pl_list = []
for pl in policy_list:
    X_pl = []
    for idx in range(len(OBS)):
        age = list(OBS[['AGE','SEX']].iloc[0])[0]
        sex = list(OBS[['AGE', 'SEX']].iloc[0])[1]
        x_pl = pl(age,sex)
        X_pl.append(x_pl)
    X_pl_list.append(X_pl)

for age in [0,1]:
    for sex in [0,1]:
        print(len(EXP[(EXP['AGE'] == age) & (EXP['SEX']==sex) & (EXP['X']==1)]))


# # ducb = DUCB(policy_list,pred_list,opt_pl,T, X_pl_list,Y_pl_list,Z)
# # Vanila D-UCB (Rajet sen, 2018)
# prob_opt_list,avg_loss_list,num_pull = ducb.conduct_DUCB()
# # Bounded D-UCB (B-DUCB) (JZ)
# bdd_prob_opt_list,bdd_avg_loss_list,bdd_num_pull = ducb.conduct_BDUCB(Bdd)