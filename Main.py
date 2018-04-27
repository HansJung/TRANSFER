from DataGen import DataGen
from DPFit import DPFit
from CausalBound import CausalBound
from UCB import UCB
import copy
import numpy as np
import matplotlib.pyplot as plt

''' From DataGen '''
D = 20
N = 10000
T = int(N/2)
Ns = 20
# seed_num = np.random.randint(10000000)
seed_num = 2917291


# 8030328 is good seed
Mode = 'crazy'

print('Seed_num', seed_num)
datagen = DataGen(D,N,Ns,Mode, seed_num)
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

iter_opt = 100
bound_list = []
bounded_models = []
weights = []
C_list = []
dpobs_list = []



for x in [0,1]:
    init_compo = 100
    Yobs_x = Obs[Obs['X']==x]['Y']
    Yintv_x = Intv[Intv['X']==x]['Y']

    DPFit_intv = DPFit(Yintv_x,init_compo)
    DPFit_intv.Conduct()
    dpintv = DPFit_intv.dpgmm

    DPFit_obs = DPFit(Yobs_x, init_compo)
    DPFit_obs.Conduct()
    dpobs = DPFit_obs.dpgmm
    init_compo = sum(1 for x in dpobs.weights_ if x > 1e-2)
    DPFit_obs = DPFit(Yobs_x, init_compo)
    DPFit_obs.Conduct()
    dpobs = DPFit_obs.dpgmm
    dpobs_list.append(dpobs)

    ''' From Causal Bound '''
    px = len(Obs[Obs['X'] == x])/ N
    C =  -np.log(px)
    C_list.append(C)
    CB = CausalBound(dpobs,C)

    # LB, UB, lower, upper = CB.Solve_Optimization(C, init_compo, iter_opt)
    # bound_list.append([LB, UB])
    # bounded_models.append([lower, upper])

    # Arbitrary density
    if Mode == 'crazy':
        LB, UB, min_mu_list, max_mu_list, min_weight_list, max_weight_list = CB.Conduct_Optimization(C, init_compo, iter_opt)
        if x == 0 :
            X0 = [LB, UB, min_mu_list, max_mu_list, min_weight_list, max_weight_list, C, dpobs, init_compo, dpintv]
        elif x == 1:
            X1 = [LB, UB, min_mu_list, max_mu_list, min_weight_list, max_weight_list, C, dpobs, init_compo, dpintv]
        bound_list.append([LB,UB])
        # bounded_models.append([lower,upper])
        # weights.append([lower_weight, upper_weight])

    # Easy density
    if Mode == 'easy':
        f_std = np.std(Yintv_x)
        f_mean = np.mean(Yintv_x)
        g_std = copy.copy(f_std)
        LB,UB = CB.Easy_bound(f_mean,f_std,g_std, C)
        bound_list.append([LB,UB])

print('')
print('Observational mean')
print('X=0',np.mean(Obs[Obs['X']==0]['Y']))
print('X=1',np.mean(Obs[Obs['X']==1]['Y']))
print('')

print(bound_list)
print(np.mean(Intv[Intv['X']==0]['Y']), np.mean(Intv[Intv['X']==1]['Y']))

# #
# ''' Test Code '''
# CB = CausalBound(dpobs,C)
# x0_min_mu_list = X0[2]
# x0_max_mu_list = X0[3]
# x1_min_mu_list = X1[2]
# x1_max_mu_list = X1[3]
#
# x0_min_weight_list = X0[4]
# x0_max_weight_list = X0[5]
# x1_min_weight_list = X1[4]
# x1_max_weight_list = X1[5]
#
# C0 = X0[6]
# C1 = X1[6]
#
# dpobs_x0 = X0[7]
# dpobs_x1 = X1[7]
#
# cls_x0 = X0[8]
# cls_x1 = X1[8]
#
# dpintv_x0 = X0[9]
# dpintv_x1 = X0[9]
#
# f_means_x0 = dpobs_x0.means_.T[0]
# f_stds_x0 = np.ndarray.flatten(np.round(np.sqrt(dpobs_x0.covariances_), 12))
# f_weights_x0 = dpobs_x0.weights_
#
# f_means_x1 = dpobs_x1.means_.T[0]
# f_stds_x1 = np.ndarray.flatten(np.round(np.sqrt(dpobs_x1.covariances_), 12))
# f_weights_x1 = dpobs_x1.weights_
#
# print('x0_lower Const', CB.KL_GMM(f_weights_x0,x0_min_weight_list[len(x0_min_weight_list)-1],
#                                   f_means_x0,x0_min_mu_list[len(x0_min_mu_list)-1],f_stds_x0,f_stds_x0), C0)
#
# print('x0_upper Const', CB.KL_GMM(f_weights_x0,x0_max_weight_list[len(x0_max_weight_list)-1],
#                                   f_means_x0,x0_max_mu_list[len(x0_max_mu_list)-1],f_stds_x0,f_stds_x0), C0)
#
# print('x1_lower Const', CB.KL_GMM(f_weights_x1,x1_min_weight_list[len(x1_min_weight_list)-1],
#                                   f_means_x1,x1_min_mu_list[len(x1_min_mu_list)-1],f_stds_x1,f_stds_x1), C1)
#
# print('x1_upper Const', CB.KL_GMM(f_weights_x1,x1_max_weight_list[len(x1_max_weight_list)-1],
#                                   f_means_x1,x1_max_mu_list[len(x1_max_mu_list)-1],f_stds_x1,f_stds_x1), C1)
#
#
#
''' From UCB '''
K = 1
ucb = UCB(bound_list,Intv, K, T)
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
#
#
# # if np.mean(Intv[Intv['X']==0]['Y']) > np.mean(Intv[Intv['X']==1]['Y']):
# #     opt_arm = 0
# # else:
# #     opt_arm = 1
# #
# # color_box = ['r','b']
# # choice_box = []
# #
# # cur_num = -1
# # for idx in range(len(UCB_list_B)):
# #     prev_num = cur_num
# #     choice_arm = Arm_B
# #     ucb_cb = max(UCB_list_B[idx])
# #     ucb_hat_chosen = min(UCB_hat_list_B[idx])
# #
# #     if max(UCB_list_B[idx]) in UCB_hat_list_B[idx]:
# #         # print('UCB')
# #         choice_box.append(1)
# #         cur_num = 1
# #
# #     else:
# #         # print('CB')
# #         choice_box.append(0)
# #         cur_num = 0
# #
# #     if idx > 0 and prev_num != cur_num:
# #         rem_num = idx
# #         rem_arm = prev_num
# #
# # #
# # # #
# # # color_list = [color_box[idx] for idx in choice_box]
# # # plt.figure()
# # # plt.title('Illustration: which bounds affect choice of arms')
# # # phase1 = plt.scatter(range(rem_num),[1]*rem_num,c=color_list[:rem_num])
# # # phase2 = plt.scatter(range(rem_num, len(choice_box)),[1]*(len(choice_box) - rem_num),c=color_list[rem_num:])
# # # plt.yticks([])
# # # if rem_arm == 0:
# # #     plt.legend([phase1, phase2],['IT-bound','UCB'])
# # # else:
# # #     plt.legend([phase1, phase2], ['IT-bound', 'CB'])
#
