import numpy as np
import copy
import itertools
import pandas as pd
from scipy.special import lambertw

class DUCB(object):
    def __init__(self, bound_list, policy_list, Z_obs,  Y_pl_list, X_pl_list, K, T):
        self.bdd_list = bound_list # List of upper and lower bounds
        self.policy_list = policy_list # List of policy models
        self.policy_idx_list = list(range(len(policy_list)))
        self.Z = Z_obs
        self.Y_list = Y_pl_list
        self.X_list = X_pl_list
        self.K = K
        self.T = T

        self.LB_exp = []
        self.UB_exp = []
        for exp_bdd in bound_list:
            self.LB_exp.append(exp_bdd[0])
            self.UB_exp.append(exp_bdd[1])
        self.l_max = max(self.LB_exp)

        self.mu_list = []
        for idx in range(len(self.Y_list)):
            self.mu_list.append(np.mean(self.Y_list[idx]))
        self.opt_exp = self.mu_list.index(max(self.mu_list))
        self.mu_opt = max(self.mu_list)

    def Pull_Receive(self, pl, at, Ns):
        return self.Y_list[pl][Ns[pl]]

    def Exp_Cut(self):
        aftercut_LB_exp = []
        aftercut_UB_exp = []
        aftercut_mu_list = []
        aftercut_policy_idx_list = []

        for sto in self.policy_idx_list:
            lb_e = self.LB_exp[sto]
            ub_e = self.UB_exp[sto]
            if ub_e < self.l_max: # CUT
                continue
            else:
                aftercut_policy_idx_list.append(sto)
                aftercut_LB_exp.append(self.LB_exp[sto])
                aftercut_UB_exp.append(self.UB_exp[sto])
                mu_sto = self.mu_list[sto]
                aftercut_mu_list.append(mu_sto)

        return aftercut_policy_idx_list

    def Compute_divergence_two(self, poly_k, poly_j, Z_obs, Xk):
        pi_k_probs = poly_k.predict_proba(Z_obs)
        pi_j_probs = poly_j.predict_proba(Z_obs)

        sum_elem = 0
        N = len(Z_obs)
        for idx in range(N):
            pi_k_idx = pi_k_probs[idx][Xk[idx]]
            pi_j_idx = pi_j_probs[idx][Xk[idx]]
            sum_elem += np.exp(pi_k_idx / (pi_j_idx + 1e-8)) - 1

        return (sum_elem / N) - 1

    def Compute_Mkj(self, poly_k, poly_j, Z_obs, Xk):
        div_kj = self.Compute_divergence_two(poly_k, poly_j, Z_obs, Xk)
        return np.log(div_kj + 1) + 1

    def Mkj_Matrix(self, policy_list, X_pl_list, Z_obs):
        N_poly = len(policy_list)
        poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
        M_mat = np.zeros((N_poly, N_poly))
        for k, j in poly_idx_iter:
            # if k != j:
            poly_k = policy_list[k]
            poly_j = policy_list[j]
            Xk = X_pl_list[k]
            M_mat[k, j] = self.Compute_Mkj(poly_k, poly_j, Z_obs, Xk)
        return M_mat

    def Poly_ratio_kj(self,poly_k, poly_j, zs, xj_s):
        '''

        :param poly_k: Policy K (pi_k)
        :param poly_j: Policy j (pi_j)
        :param zs: Numeric. Context at time s
        :param xj_s: Numeric. Arm pulled by pi_j at time s
        :return:
        '''
        # Compute pi_k(x_j(s) | zs) / pi_j(x_j(s)|zs)
        ## pi_k(x_j(s)|zs)
        ### Probability of policy k choose X_j(s) given zs
        pi_k = poly_k.predict_proba(pd.DataFrame([zs]))[0][xj_s]
        pi_j = poly_j.predict_proba(pd.DataFrame([zs]))[0][xj_s]
        # try:
        #     pi_k = poly_k.predict_proba(zs)[0][xj_s]
        # except:
        #     pi_k = poly_k.predict_proba(pd.DataFrame([zs]))[xj_s]
        #
        # ### Probability of policy j choose X_j(s) given zs
        # try:
        #     pi_j = poly_j.predict_proba(zs)[0][xj_s]
        # except:
        #     pi_j = poly_j.predict_proba(pd.DataFrame([zs]))[xj_s]

        return pi_k / (pi_j+1e-8)

    def Clipped_est(self, k, Ns, M, policy_idx_list, policy_list, Tau_s, Y_pl_list, Z_obs, X_pl_list, t):
        '''

        :param k: Int. index number of policy
        :param Ns: Dict. of number of choices of the policy
        :param M: Matrix of entropy between each policy
        :param policy_idx_list: List of indices of policy
        :param policy_list: List of policy
        :param Tau_s: Dict. of the list of time-steps when the expert S was chosen
        :param Y_pl_list: List of Y resulting from pl's arm pulling
        :param Z_obs: List of context Z_t
        :param X_pl_list: List of arm pulling of pl.
        :param t: Current time t
        :return:
        '''

        eps_t = 2 / t # Unbiased error reduced over time t
        poly_k = policy_list[k] # Policy K of our interest

        # Z_k(t) term = sum_{j} n_j(t)/M_{kj} for each policy j
        Zk_t = 0
        for j in policy_idx_list:
            Zk_t += Ns[j] / M[k, j]

        # Clipped estimator for policy k
        mu_k = 0
        for j in policy_idx_list: # Sum over policy j = 1 to N
            poly_j = policy_list[j] # For each policy j
            Xj = X_pl_list[j] # j'th arm pulling
            # For each time s in Tau_s[j]
            # Each time s where the policy j was selected
            for s in Tau_s[j]:
                # If Poly ratio less than 2log(2/eps(t))M_kj
                if self.Poly_ratio_kj(poly_k, poly_j, Z_obs.ix[s], Xj[s]) < 2 * np.log(2 / eps_t) * M[k, j]:
                    # 1/Mkj Y_j(s) * poly ratio
                    mu_k += (1 / M[k, j]) * (Y_pl_list[j][s]) * self.Poly_ratio_kj(poly_k, poly_j, Z_obs.ix[s], Xj[s]) 
                else:
                    mu_k += 0
        mu_k = mu_k / Zk_t
        return mu_k

    def Compute_Akj(self, t, k, j, M, X_pl_list, Y_pl_list, Z_obs, policy_list):
        '''

        :param t: Time t
        :param k: Index of policy k
        :param j: Index of policy j
        :param M: Policy matrix M
        :param X_pl_list: List of List of arm pull of policy
        :param Y_pl_list: List of List of arm pull of reward
        :param Z_obs: Observation at Z_obs
        :param policy_list:
        :return:
        '''

        Mkj = M[k,j] # Policy matrix
        eps_t = 2 / (t+1e-8)  # Unbiased error reduced over time t
        Zt = Z_obs.ix[t] # Context at time t
        Xjt = X_pl_list[j][t]
        Yjt = Y_pl_list[j][t]

        poly_k = policy_list[k]  # Policy K of our interest
        poly_j = policy_list[j]

        poly_ratio_kj = self.Poly_ratio_kj(poly_k, poly_j, Zt, Xjt)
        idx_poly = poly_ratio_kj < 2 * np.log(2 / eps_t) * M[k, j]

        Akj = idx_poly * poly_ratio_kj * Yjt * (1/Mkj)
        return Akj




    def Upper_bonus(self, k, Ns, M, policy_idx_list, t):
        Zk_t = 0
        c1 = 16
        for j in policy_idx_list:
            Zk_t += Ns[j] / M[k, j]
        C = (np.sqrt(c1 * t * np.log(t))) / (Zk_t)
        Bt = np.real(C * lambertw(2 / (C + 1e-8)))
        Sk = 1.5 * Bt
        return Sk


    def DUCB(self):
        Ns = dict()
        Tau_s = dict()
        sum_reward = 0
        cum_regret = 0
        Reward_pl = dict()

        prob_opt_list = []
        cum_regret_list = []
        Sto_pick = []

        Akjt_dict = dict()
        for i in self.policy_idx_list:
            for j in self.policy_idx_list:
                Akjt_dict[i,j] = list()

        mu_k_dict = dict()
        for i in self.policy_idx_list:
            mu_k_dict[i] = list()


        M = self.Mkj_Matrix(self.policy_list,self.X_list,self.Z)

        for s in self.policy_idx_list:
            Ns[s] = 0
            Tau_s[s] = []
            Reward_pl[s] = []

        # Initial pulling
        for t in range(self.K * len(self.policy_idx_list)):
            print(t)
            # Policy pick!
            st = np.mod(t, len(self.policy_idx_list))

            # Policy choosing arm
            ## Expert st's arm t'th arm choice
            at = self.X_list[st][t]
            rt = self.Y_list[st][t]

            # Store the time step when the expert st chose.
            Tau_s[st].append(t)
            Ns[st] += 1
            Sto_pick.append(st)

            Reward_pl[st].append(rt)
            sum_reward += rt

            prob_opt = Ns[self.opt_exp] / (t + 1)
            cum_regret += self.mu_opt - self.mu_list[st]

            prob_opt_list.append(prob_opt)
            cum_regret_list.append(cum_regret)

        # Initial Compute Akj
        for t in range(self.K * len(self.policy_idx_list)):
            for k in self.policy_idx_list:
                for j in self.policy_idx_list:
                    Akj = self.Compute_Akj(t,k,j,M,self.X_list,self.Y_list,self.Z,self.policy_list)
                    Akjt_dict[k,j].append(Akj)


        # Run
        for t in range(self.K * len(self.policy_idx_list), self.T):
            # print(t)
            for k in self.policy_idx_list:
                for j in self.policy_idx_list:
                    Akj = self.Compute_Akj(t,k,j,M,self.X_list,self.Y_list,self.Z,self.policy_list)
                    Akjt_dict[k,j].append(Akj)

            mu_k_list = []
            UCB_list = []
            for k in self.policy_idx_list:
                mu_k_t = 0
                zk = 0
                for j in self.policy_idx_list:
                    for s in Tau_s[j]:
                        mu_k_t += Akjt_dict[k,j][s]
                    zk += Ns[j] / M[k,j]
                mu_k_t /= zk
                # mu_k_dict[k].append(mu_k_t)
                mu_k_list.append(mu_k_t)
                s_k = self.Upper_bonus(k, Ns, M, self.policy_idx_list, t)
                UCB_list.append(mu_k_t + s_k)
                mu_k_dict[k].append(mu_k_t+s_k)
            # Choose the expert and store the expert index
            k_star = np.argmax(UCB_list)
            Sto_pick.append(k_star)

            # Policy k_star's  choosing arm
            at = self.X_list[k_star][t]

            # Poliy k_star receiving reward
            rt = self.Y_list[k_star][t]

            # Store the time step when the expert st chose.
            Tau_s[k_star].append(t)
            Ns[k_star] += 1
            Sto_pick.append(k_star)

            Reward_pl[k_star].append(rt)
            sum_reward += rt

            prob_opt = Ns[self.opt_exp] / (t + 1)
            cum_regret += self.mu_opt - self.mu_list[k_star]

            prob_opt_list.append(prob_opt)
            cum_regret_list.append(cum_regret)
        return prob_opt_list, cum_regret_list, Sto_pick, mu_k_dict

    def B_DUCB(self):
        Ns = dict()
        Tau_s = dict()
        sum_reward = 0
        cum_regret = 0
        Reward_pl = dict()

        prob_opt_list = []
        cum_regret_list = []
        Sto_pick = []

        Akjt_dict = dict()
        for i in self.policy_idx_list:
            for j in self.policy_idx_list:
                Akjt_dict[i, j] = list()

        mu_k_dict = dict()
        for i in self.policy_idx_list:
            mu_k_dict[i] = list()

        M = self.Mkj_Matrix(self.policy_list, self.X_list, self.Z)

        for s in self.policy_idx_list:
            Ns[s] = 0
            Tau_s[s] = []
            Reward_pl[s] = []

        # Initial pulling
        for t in range(self.K * len(self.policy_idx_list)):
            print(t)
            # Policy pick!
            st = np.mod(t, len(self.policy_idx_list))

            # Policy choosing arm
            ## Expert st's arm t'th arm choice
            at = self.X_list[st][t]
            rt = self.Y_list[st][t]

            # Store the time step when the expert st chose.
            Tau_s[st].append(t)
            Ns[st] += 1
            Sto_pick.append(st)

            Reward_pl[st].append(rt)
            sum_reward += rt

            prob_opt = Ns[self.opt_exp] / (t + 1)
            cum_regret += self.mu_opt - self.mu_list[st]

            prob_opt_list.append(prob_opt)
            cum_regret_list.append(cum_regret)

        # Initial Compute Akj
        for t in range(self.K * len(self.policy_idx_list)):
            for k in self.policy_idx_list:
                for j in self.policy_idx_list:
                    Akj = self.Compute_Akj(t, k, j, M, self.X_list, self.Y_list, self.Z, self.policy_list)
                    Akjt_dict[k, j].append(Akj)

        # Run
        for t in range(self.K * len(self.policy_idx_list), self.T):
            # print(t)
            for k in self.policy_idx_list:
                for j in self.policy_idx_list:
                    Akj = self.Compute_Akj(t, k, j, M, self.X_list, self.Y_list, self.Z, self.policy_list)
                    Akjt_dict[k, j].append(Akj)

            mu_k_list = []
            UCB_list = []
            for k in self.policy_idx_list:
                mu_k_t = 0
                zk = 0
                for j in self.policy_idx_list:
                    for s in Tau_s[j]:
                        mu_k_t += Akjt_dict[k, j][s]
                    zk += Ns[j] / M[k, j]
                mu_k_t /= zk
                mu_k_dict[k].append(mu_k_t)
                mu_k_list.append(mu_k_t)
                s_k = self.Upper_bonus(k, Ns, M, self.policy_idx_list, t)
                UCB_k = mu_k_t + s_k
                UB_k = self.UB_exp[k]
                UCB_list.append(min(UCB_k,UB_k))
            # Choose the expert and store the expert index
            k_star = np.argmax(UCB_list)
            Sto_pick.append(k_star)

            # Policy k_star's  choosing arm
            at = self.X_list[k_star][t]

            # Poliy k_star receiving reward
            rt = self.Y_list[k_star][t]

            # Store the time step when the expert st chose.
            Tau_s[k_star].append(t)
            Ns[k_star] += 1
            Sto_pick.append(k_star)

            Reward_pl[k_star].append(rt)
            sum_reward += rt

            prob_opt = Ns[self.opt_exp] / (t + 1)
            cum_regret += self.mu_opt - self.mu_list[k_star]

            prob_opt_list.append(prob_opt)
            cum_regret_list.append(cum_regret)
        return prob_opt_list, cum_regret_list, Sto_pick, mu_k_dict

    def Bandit_Run(self):
        prob_opt_list, cum_regret_list, Sto_pick, mu_k_dict = self.DUCB()
        prob_opt_list_B, cum_regret_list_B, Sto_pick_B, mu_k_dict_B = self.B_DUCB()
        return [[prob_opt_list, cum_regret_list, Sto_pick, mu_k_dict],[prob_opt_list_B, cum_regret_list_B, Sto_pick_B, mu_k_dict_B]]



            #
            #
            # print(t)
            # # Observe context
            # zt = self.Z.ix[t]
            #
            # # Store the UCB of each policy
            # UCB_list = []
            # # Compute K(t) = argmax_k Uk(t-1)
            # ## Compute clip estimator
            # for k in self.policy_idx_list: # For each stochastic policy
            #     st = self.policy_list[k]
            #     # Compute expert k's clipped estimator
            #     mu_k = self.Clipped_est(k,Ns,M,self.policy_idx_list,self.policy_list,Tau_s,self.Y_list,self.Z,self.X_list,t)
            #     # Compute expert k'th upper bound
            #     s_k = self.Upper_bonus(k,Ns,M,self.policy_idx_list,t)
            #     UCB_list.append(mu_k + s_k)

        #     # Choose the expert and store the expert index
        #     k_star = np.argmax(UCB_list)
        #     Sto_pick.append(k_star)
        #
        #     # Policy k_star's  choosing arm
        #     at = self.X_list[k_star][t]
        #
        #     # Poliy k_star receiving reward
        #     rt = self.Y_list[k_star][t]
        #
        #     # Store the time step when the expert st chose.
        #     Tau_s[k_star].append(t)
        #     Ns[k_star] += 1
        #     Sto_pick.append(k_star)
        #
        #     Reward_pl[k_star].append(rt)
        #     sum_reward += rt
        #
        #     prob_opt = Ns[self.opt_exp] / (t + 1)
        #     cum_regret += self.mu_opt - self.mu_list[k_star]
        #
        #     prob_opt_list.append(prob_opt)
        #     cum_regret_list.append(cum_regret)
        # return prob_opt_list, cum_regret_list











    # def pairwise(self,iterable):
    #     "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    #     a = iter(iterable)
    #     return zip(a, a)
    #
    # def MedianofMeans(self, A, ngroups=55):
    #     '''Calculates median of means given an Array A, ngroups is the number of groups '''
    #     if ngroups > len(A):
    #         ngroups = len(A)
    #     Aperm = np.random.permutation(A)
    #     try:
    #         lists = np.split(Aperm, ngroups)
    #     except:
    #         x = len(A) / ngroups
    #         ng = len(A) / (x + 1)
    #         Atrunc = Aperm[0:ng * x]
    #         lists = np.split(Atrunc, ng) + [Aperm[ng * x::]]
    #     plist = pd.Series(lists)
    #     mlist = plist.apply(np.mean)
    #     return np.median(mlist)
    #
    # def pairmodel_divergence(self, Z, M1, M2, option=1):
    #     '''Function to calculate divergence between two experts
    #     1. When Xtrain is a data-set of features/contexts and M1,M2 are models (eg. xgboost) then we get predict_proba predictions on Xtrain with M1,
    #     M2 and then calculate divergence using the probability values.
    #     2. When Xtrain is none and the probability values themselves are supplied as M1, M2 then the divergence is calculated directly
    #     '''
    #     if Z is None:
    #         P1 = M1
    #         P2 = M2
    #     else:
    #         P1 = M1.predict_proba(Z)
    #         P2 = M2.predict_proba(Z)
    #
    #     if option == 1:
    #         X = P1 / P2
    #         fx = X * np.exp(X - 1) - 1
    #         Efx = P2 * fx
    #         Efx_sum = np.sum(Efx, axis=1)
    #         l = np.where(Efx_sum > 1e100)
    #         inan = np.where(np.isnan(Efx_sum) == True)
    #         Efx_sum[inan] = 1e100
    #         Efx_sum[l] = 1e100
    #         return self.MedianofMeans(Efx_sum)
    #     if option == 2:
    #         X = P1 / P2
    #         fx = X ** 2 - 1
    #         Efx = P2 * fx
    #         Efx_sum = np.sum(Efx, axis=1)
    #         inan = np.where(np.isnan(Efx_sum) == True)
    #         Efx_sum[inan] = 1e10
    #         l = np.where(Efx_sum > 1e10)
    #         Efx_sum[l] = 1e10
    #         return self.MedianofMeans(Efx_sum)








    #
    #
    # def UCB(self, arm_list,u_list, K):
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
    #             print(t)
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
    #             prob_opt = Na_T[self.opt_exp] / (t + 1)
    #             cum_regret += self.mu_opt - u_list[at]
    #
    #             prob_opt_list.append(prob_opt)
    #             cum_regret_list.append(cum_regret)
    #         return prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T
    #
    # def B_UCB(self, K):
    #     arm_list, LB_arm, UB_arm, u_list = self.Arm_Cut()
    #
    #     if len(arm_list) == 1:
    #         print("All cut")
    #         return arm_list
    #     else:
    #         if u_list[0] > u_list[1]:
    #             opt_arm = 0
    #             u_opt = u_list[0]
    #         else:
    #             opt_arm = 1
    #             u_opt = u_list[1]
    #
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
    #         for t in range(K * len(arm_list)):
    #             # Pulling!
    #             at = np.mod(t, len(arm_list))
    #             Arm.append(at)
    #             rt = self.Pull_Receive(at, Na_T)
    #             Reward_arm[at].append(rt)
    #             sum_reward += rt
    #             Na_T[at] += 1
    #             prob_opt = Na_T[opt_arm] / (t + 1)
    #             cum_regret += u_opt - u_list[at]
    #
    #             prob_opt_list.append(prob_opt)
    #             cum_regret_list.append(cum_regret)
    #
    #         # Run!
    #         UCB_list = []
    #         UCB_hat_list = []
    #         X_hat_list = []
    #         what_choose = []
    #         for t in range(K * len(arm_list), self.T):
    #             X_hat_arm = []
    #             UB_list = []
    #             UCB_hat = []
    #             for a in arm_list:
    #                 # standard UCB
    #                 x_hat = np.mean(Reward_arm[a])
    #                 X_hat_arm.append(x_hat)
    #                 upper_a = np.sqrt((3 * np.log(t)) / (2 * Na_T[a]))
    #                 UCB_a = x_hat + upper_a
    #                 UCB_hat.append(UCB_a)
    #                 UB_a = min(UCB_a, UB_arm[a])
    #                 UB_list.append(UB_a)
    #             UCB_list.append(UB_list)
    #             UCB_hat_list.append(UCB_hat)
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
    #             prob_opt = Na_T[opt_arm] / (t + 1)
    #             cum_regret += u_opt - u_list[at]
    #
    #             prob_opt_list.append(prob_opt)
    #             cum_regret_list.append(cum_regret)
    #         return prob_opt_list, cum_regret_list, UCB_list, UCB_hat_list, Arm, X_hat_list, Na_T
    #
    # def Bandit_Run(self):
    #     prob_opt, cum_regret, UCB_list, Arm, X_hat_list,  Na_T = self.UCB(self.arm_list,self.u_list,self.K)
    #     prob_opt_B, cum_regret_B, UCB_list_B, UCB_hat_list_B, Arm_B,X_hat_list_B, Na_T_B = self.B_UCB(self.K)
    #     return [[prob_opt, cum_regret, UCB_list, Arm,X_hat_list, Na_T],[prob_opt_B, cum_regret_B, UCB_list_B, UCB_hat_list_B, Arm_B, X_hat_list_B, Na_T_B]]


