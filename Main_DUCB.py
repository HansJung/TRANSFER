import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataGen_DUCB import DataGen
from StoExp import StoExp
from DPFit import DPFit
from CausalBound import CausalBound
import itertools
from scipy.special import lambertw


D = 5
N = 1000
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
N_poly = len(policy_list)

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

def Compute_divergence_two(poly_k, poly_j, Z_obs, Xk):
    pi_k_probs = poly_k.predict_proba(Z_obs)
    pi_j_probs = poly_j.predict_proba(Z_obs)

    sum_elem = 0
    N = len(Z_obs)
    for idx in range(N):
        pi_k_idx = pi_k_probs[idx][Xk[idx]]
        pi_j_idx = pi_j_probs[idx][Xk[idx]]
        sum_elem += np.exp(pi_k_idx / (pi_j_idx + 1e-8)) - 1

    return (sum_elem/N)-1

def Compute_Mkj(poly_k, poly_j, Z_obs, Xk):
    div_kj = Compute_divergence_two(poly_k,poly_j,Z_obs,Xk)
    return np.log(div_kj + 1)+1

def Mkj_Matrix(policy_list, X_pl_list, Z_obs):
    N_poly = len(policy_list)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        poly_k = policy_list[k]
        poly_j = policy_list[j]
        Xk = X_pl_list[k]
        M_mat[k, j] = Compute_Mkj(poly_k, poly_j, Z_obs, Xk)
    return M_mat

def Poly_ratio_kj(poly_k, poly_j, zs, xj_s):
    return poly_k.predict_proba(zs)[xj_s] / poly_j.predict_proba(zs)[xj_s]

def Clipped_est(k, Ns, M, policy_idx_list, policy_list, Tau_s, Y_pl_list, Z_obs, X_pl_list, t):
    eps_t = 2/t
    poly_k = policy_list[k]
    Zk_t = 0
    for j in policy_idx_list:
        Zk_t += Ns[j] / M[k,j]

    mu_k = 0
    for j in policy_idx_list:
        poly_j = policy_list[j]
        Xj = X_pl_list[j]
        for s in Tau_s[j]:
            if Poly_ratio_kj(poly_k, poly_j, Z_obs.ix[s], Xj[s]) < 2*np.log(2/eps_t)*M[k,j]:
                mu_k += (1/M[k,j]) * (Y_pl_list[j][s]) * Poly_ratio_kj(poly_k, poly_j, Z_obs.ix[s], Xj[s])
            else:
                continue
    mu_k = mu_k / Zk_t
    return mu_k

def Upper_bonus(k, Ns, M, policy_idx_list, t):
    Zk_t = 0
    c1 = 16
    for j in policy_idx_list:
        Zk_t += Ns[j] / M[k, j]
    C = (np.sqrt(c1*t*np.log(t))) / (Zk_t)
    Bt = np.real(C*lambertw(2/(C+1e-8)))
    Sk = 1.5*Bt
    return Sk






Ns = dict()
Tau_s = dict()
for s in range(len(policy_list)):
    Ns[s] = 0
    Tau_s = []



            #
# def UCB(sto_list, u_list, K):
#     # LB_arm = self.LB_arm
#     # UB_arm = self.UB_arm
#     # arm_list = self.arm_list
#     # u_list = self.u_list
#
#     # arm_list, LB_arm, UB_arm, u_list = self.Arm_Cut()
#
#     if len(arm_list) == 1:
#         print("All cut")
#         return arm_list
#     else:
#         Arm = []
#         Na_T = dict()
#         sum_reward = 0
#         cum_regret = 0
#         Reward_arm = dict()
#
#         prob_opt_list = []
#         cum_regret_list = []
#
#         # Initial variable setting
#         for t in range(len(arm_list)):
#             # Before pulling
#             at = t
#             Na_T[at] = 0
#             Reward_arm[at] = []
#
#         # Initial pulling
#         for t in range(K*len(arm_list)):
#             # Pulling!
#             at = np.mod(t, len(arm_list))
#             Arm.append(at)
#             rt = self.Pull_Receive(at, Na_T)
#             Reward_arm[at].append(rt)
#             sum_reward += rt
#             Na_T[at] += 1
#             prob_opt = Na_T[self.opt_arm] / (t + 1)
#             cum_regret += self.u_opt - u_list[at]
#
#             prob_opt_list.append(prob_opt)
#             cum_regret_list.append(cum_regret)
#
#         # Run!
#         UCB_list = []
#         X_hat_list = []
#         for t in range(K*len(arm_list), self.T):
#             UB_list = []
#             X_hat_arm = []
#             for a in arm_list:
#                 # standard UCB
#                 x_hat = np.mean(Reward_arm[a])
#                 X_hat_arm.append(x_hat)
#                 upper_a = np.sqrt( (3 * np.log(t)) / (2 * Na_T[a]))
#                 UB_a = x_hat + upper_a
#                 UB_list.append(UB_a)
#             UCB_list.append(UB_list)
#             X_hat_list.append(X_hat_arm)
#
#             at = UB_list.index(max(UB_list))
#             Arm.append(at)
#             rt = self.Pull_Receive(at, Na_T)
#
#             Reward_arm[at].append(rt)
#             sum_reward += rt
#
#             Na_T[at] += 1
#             prob_opt = Na_T[self.opt_arm] / (t + 1)
#             cum_regret += self.u_opt - u_list[at]
#
#             prob_opt_list.append(prob_opt)
#             cum_regret_list.append(cum_regret)
#         return prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T