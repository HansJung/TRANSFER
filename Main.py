import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from DataGen import DataGen
from StoExp import StoExp
from JZ_bound import JZ_bound
from scipy.special import lambertw


# DUCB function
## Compute M
def Compute_divergence_two(poly_k, poly_j, Z, Xk):
    pi_k_probs = poly_k.predict_proba(Z)
    pi_j_probs = poly_j.predict_proba(Z)

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

def Compute_Mkj(poly_k, poly_j, Z, Xk):
    div_kj = Compute_divergence_two(poly_k, poly_j, Z, Xk)
    return np.log(div_kj + 1) + 1

def Matrix_M(policy_list, X_pl_list, Z):
    N_poly = len(policy_list)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        poly_k = policy_list[k]
        poly_j = policy_list[j]
        Xk = X_pl_list[k]
        M_mat[k, j] = Compute_Mkj(poly_k, poly_j, Z, Xk)
    return M_mat

## Compute val
def indicator(poly_ratio, eps_t, Mkj):
    ub = 2*np.log(2/eps_t) * Mkj
    if poly_ratio < ub:
        return 1
    else:
        return 0

def est_predict_proba(poly, zs, xj_s):
    try:
        result = poly.predict_proba(zs)[0][xj_s]
    except:
        zs = pd.DataFrame(np.matrix(zs))
        result = poly.predict_proba(zs)[0][xj_s]
    return result


def Poly_ratio_kj(poly_k, poly_j, zs, xj_s):
    pi_k = est_predict_proba(poly_k,zs,xj_s)
    pi_j = est_predict_proba(poly_j,zs,xj_s)
    return pi_k / (pi_j+1e-8)

def compute_val(k, j, s, eps_t, param_set):
    policy_list, pred_list, Mmat, X_pl_list, Y_pl_list, Z = param_set
    poly_k = policy_list[k]
    poly_j = policy_list[j]
    Mkj = Mmat[k,j]

    Xj = X_pl_list[j]
    Yj = Y_pl_list[j]

    zs = Z.iloc[s]
    xjs = int(Xj[s])
    yjs = Yj.iloc[s]

    poly_ratio_kj = Poly_ratio_kj(poly_k,poly_j,zs,xjs)
    return (1/Mkj) * yjs * poly_ratio_kj * indicator(poly_ratio_kj, eps_t, Mkj)

# Compute upper bonus
def upper_bonus(t, k, G, c1 = 16):
    Gk = G[k]
    return np.sqrt(c1*t*np.log(t)) / Gk






##########################################

# Parameter configuration
D = 5
N = 10000
T = 5000
seed_num = np.random.randint(10000000)
# seed_num = 1482921


# Generating Observation data
datagen = DataGen(D,N,seed_num)
OBS = datagen.obs_data()

# Externally policy
stoexp = StoExp(D)
JZ = JZ_bound()
[X,Y,Z] = JZ.sepOBS(OBS,D)

obslogit = stoexp.Logit(X,Z)
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


# DUCB part
## Initial (t=K)
N_poly = len(policy_list)
Mmat = Matrix_M(policy_list,X_pl_list,Z)
param_set = [policy_list, pred_list, Mmat, X_pl_list, Y_pl_list, Z]

dictV = dict()
sumV = dict()
G = dict()
u = dict()
num_pull = dict()
arm_stored = list()

for k in range(len(policy_list)):
    dictV[k] = [0]*len(policy_list)
    sumV[k] = 0
    G[k] = 0
    u[k] = 0
    num_pull[k] = 0

# Initial running
s = 1
for k in range(len(policy_list)):
    for j in range(len(policy_list)):
        dictV[k][j] = compute_val(k,j,s,1,param_set)
        sumV[k] += dictV[k][j]
        G[k] += 1/Mmat[k,j]
    sumV[k] = sumV[k]/G[k]
    sk = upper_bonus(1,k,G)
    u[k] = sumV[k] + sk
a_prev = np.argmax(list(u.values()))
arm_stored.append(a_prev)
num_pull[a_prev] += 1

for t in range(len(policy_list)+1, T):
    eps_t = 2/t
    for k in range(len(policy_list)):
        for j in range(len(policy_list)):
            Mkj = Mmat[k,j]
            if j == a_prev:
                dictV[k][j] += compute_val(k,j,t-1,eps_t,param_set)
                G[k] += 1/Mkj
            sumV[k] += dictV[k][j]
        sk = upper_bonus(t,k,G)
        sumV[k] = sumV[k]/G[k]
        u[k] = sumV[k] + sk
    a_prev = np.argmax(list(u.values()))
    arm_stored.append(a_prev)
    num_pull[a_prev] += 1



# combi_pl = list(itertools.product(list(range(N_poly)), list(range(N_poly))))



# for t in range(N_poly, 20):
#     eps_t = 2/t
#
#     # For each k and j
#     prev_V_list = dict()
#     max_prev_V = []
#     for k in range(N_poly):
#         prev_V_list[k] = []
#
#     for k in range(N_poly):
#         for j in range(N_poly):
#             C = 2 * np.log(2 / eps_t) * Mmat[k,j]
#             prev_V_list[k].append(compute_val(s=1,pidx_j = j, pidx_k=k,param_set=param_set, C=C))
#
#     for k in range(N_poly):
#         max_prev_V.append(max(prev_V_list[k]))
#     prev_pl = np.argmax(max_prev_V)
#     prev_V = max(max_prev_V)
#
#     t += 1
#     print(compute_V(t-1,prev_V,1,1,prev_pl,param_set, C))
#
#     # prev_V = max(prev_V_list)
#     # compute_V(t,prev_V,j,k,)