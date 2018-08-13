import pandas as pd
import numpy as np
from DUCB import DUCB
import GenData_IST
import matplotlib.pyplot as plt

X = 'RXASP'
K = 2
T = 5500

EXP,OBS = GenData_IST.RunGenData()
print(GenData_IST.QualityCheck(EXP,OBS,X))
print(GenData_IST.ObsEffect(EXP,'Y'))
print(GenData_IST.ObsEffect(OBS,'Y'))

LB,HB = GenData_IST.ComputeBound(OBS,X)
lx0,lx1 = LB
hx0,hx1 = HB
bound_list = [[lx0,hx0],[lx1,hx1]]

EXP = GenData_IST.ChangeRXASPtoX(EXP,idx_X=2)
OBS = GenData_IST.ChangeRXASPtoX(OBS,idx_X=2)



# DUCB
policy_list = []

pl1 = lambda age, sex: 1 if (age == 0) and (sex == 0) else 0
pl2 = lambda age, sex: 1 if (age == 0) and (sex == 1) else 0
pl3 = lambda age, sex: 1 if (age == 1) and (sex == 0) else 0
pl4 = lambda age, sex: 1 if (age == 1) and (sex == 1) else 0

policy_list = [pl1,pl2,pl3,pl4]

# Declaration of variables
## Store Number of Policy
NumPl = dict()
for idx in range(len(policy_list)):
    NumPl[idx] = 0

## Store Number of Arm
NumArm = dict()
for idx in range(len(pd.unique(EXP['X']))):
    NumArm[idx] = 0

PolyIdx = list()

PolyIdxDict = dict()
for idx in range(len(policy_list)):
    PolyIdxDict[idx] = list()

# DUCB matrix
import itertools
def predict_proba(pl,Z):
    probs = list()
    for idx in range(len(Z)):
        z = list(Z.iloc[idx])
        prob = pl(z[0],z[1])
        probs.append([1-prob,prob])
    return probs

def Compute_divergence_two(poly_k, poly_j, Z, Xk):
    pi_k_probs = predict_proba(poly_k,Z)
    pi_j_probs = predict_proba(poly_j,Z)

    sum_elem = 0
    N = len(Z)
    eps = 1e-8

    for idx in range(N):
        Xk_i = int(Xk[idx])
        pi_k_idx = pi_k_probs[idx][Xk_i]
        pi_j_idx = pi_j_probs[idx][Xk_i]
        div_val = pi_k_idx/(pi_j_idx + eps)
        sum_elem += div_val * np.exp(div_val - 1) - 1
    return (sum_elem / N)

def Compute_Mkj(self,poly_k, poly_j, Z, Xk):
    div_kj = self.Compute_divergence_two(poly_k, poly_j, Z, Xk)
    return np.log(div_kj + 1) + 1

def Matrix_M(self, policy_list, X_pl_list, Z):
    N_poly = len(policy_list)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        poly_k = policy_list[k]
        poly_j = policy_list[j]
        Xk = X_pl_list[k]
        M_mat[k, j] = self.Compute_Mkj(poly_k, poly_j, Z, Xk)
    return M_mat

def PolyEst(pl,Z): # Assume dim(Z) = 2
    return pl(Z[0],Z[1])

def PolyProb(pl,X,Z):
    pl_est = PolyEst(pl,Z)
    if X == pl_est:
        return 1
    else:
        return 0


# At t = 0
t = 0

## Observe Z
Z = list(EXP[['AGE','SEX']].iloc[t])

## Randomly choose policy
policy_index = np.random.randint(len(policy_list))
pl = policy_list[policy_index]

## Update data structure
NumPl[policy_index] += 1
PolyIdx.append(policy_index)
PolyIdxDict[policy_index].append(t)

## Chosen policy pull arm
Xpl = pl(Z[0],Z[1])
## Update NumArm
NumArm[Xpl] += 1
## Receive the reward.
Ypl = EXP[EXP['X']==Xpl]['Y'].iloc[NumArm[Xpl]-1]

########################################################################
# At t = 1
t += 1

## Observe Z
Z = list(EXP[['AGE','SEX']].iloc[t])

## Divide the group
### Number of groups
# lt = NumGroup(t)
