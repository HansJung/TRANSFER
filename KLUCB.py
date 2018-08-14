import numpy as np
import copy
from scipy.optimize import minimize

class KLUCB(object):
    def __init__(self, bound_list, Intv, K, T):
        self.K = K
        self.bdd_list = bound_list
        self.T = T
        self.arm_list = list(range(len(bound_list)))

        self.LB_arm = []
        self.UB_arm = []
        for arm_bdd in bound_list:
            self.LB_arm.append(arm_bdd[0])
            self.UB_arm.append(arm_bdd[1])
        self.l_max = max(self.LB_arm)

        u0 = np.mean(Intv[Intv['X'] == 0]['Y'])
        u1 = np.mean(Intv[Intv['X'] == 1]['Y'])
        if u0 < u1:
            u_opt = u1
            opt_arm = 1
        else:
            u_opt = u0
            opt_arm = 0

        self.u_opt = u_opt
        self.opt_arm = opt_arm
        self.Intv = Intv
        self.u_list = [u0,u1]

    def Arm_Cut(self):
        new_arm = []
        LB_arm = []
        UB_arm = []
        u_list = []

        for a in self.arm_list:
            lb_a = self.LB_arm[a]
            ub_a = self.UB_arm[a]
            if ub_a < self.l_max:
                continue
            else:
                new_arm.append(a)
                LB_arm.append(self.LB_arm[a])
                UB_arm.append(self.UB_arm[a])
                ua = np.mean(self.Intv[self.Intv['X'] == a]['Y'])
                u_list.append(ua)

        return new_arm, LB_arm, UB_arm, u_list


    def Pull_Receive(self, at, Na_T):
        return list(self.Intv[self.Intv['X'] == at]['Y'])[Na_T[at]]

    def BinoKL(self, mu_hat, mu):
        return mu_hat * np.log(mu_hat / mu) + (1 - mu_hat) * np.log((1 - mu_hat) / (1 - mu))


    def MaxKL(self, mu_hat, ft, NaT, init_maxval=1):
        maxval = copy.copy(init_maxval)
        mu = copy.copy(mu_hat)
        M = ft/NaT

        terminal_cond = 1e-8
        while 1:
            mu_cand = (mu + maxval) / 2
            KL_val = self.BinoKL(mu_hat, mu_cand)
            diff = np.abs(KL_val - M)
            if KL_val < M:
                if diff < terminal_cond:
                    mu = mu_cand
                    return mu
                else:
                    mu = copy.copy(mu_cand)
            else:
                maxval = copy.copy(mu_cand)


    def KLUCB(self, arm_list,u_list, K):
        # LB_arm = self.LB_arm
        # UB_arm = self.UB_arm
        # arm_list = self.arm_list
        # u_list = self.u_list

        # arm_list, LB_arm, UB_arm, u_list = self.Arm_Cut()

        ft = lambda x: np.log(x) + 3 * np.log(np.log(x))
        if len(arm_list) == 1:
            print("All cut")
            return arm_list
        else:
            Arm = []
            Na_T = dict()
            sum_reward = 0
            cum_regret = 0
            Reward_arm = dict()

            prob_opt_list = []
            cum_regret_list = []

            # Initial variable setting
            for t in range(len(arm_list)):
                # Before pulling
                at = t
                Na_T[at] = 0
                Reward_arm[at] = []

            # Initial pulling
            for t in range(K*len(arm_list)):
                # Pulling!
                at = np.mod(t, len(arm_list))
                Arm.append(at)
                rt = self.Pull_Receive(at, Na_T)
                Reward_arm[at].append(rt)
                sum_reward += rt
                Na_T[at] += 1
                prob_opt = Na_T[self.opt_arm] / (t + 1)
                cum_regret += self.u_opt - u_list[at]

                prob_opt_list.append(prob_opt)
                cum_regret_list.append(cum_regret)

            # Run!
            UCB_list = []

            # data_collection = {'Arm':Arm, 'Reward':Reward, 'Cum_regret': cum_regret, 'Prob_opt':prob_opt }
            X_hat_list = []
            for t in range(K*len(arm_list), self.T):
                UB_list = []
                X_hat_arm = []
                for a in arm_list:
                    # standard UCB
                    mu_hat = np.mean(Reward_arm[a])
                    UB_a = self.MaxKL(mu_hat, ft(t), Na_T[a], init_maxval=1)
                    UB_list.append(UB_a)
                UCB_list.append(UB_list)

                at = UB_list.index(max(UB_list))
                Arm.append(at)
                rt = self.Pull_Receive(at, Na_T)

                Reward_arm[at].append(rt)
                sum_reward += rt

                Na_T[at] += 1
                prob_opt = Na_T[self.opt_arm] / (t + 1)
                cum_regret += self.u_opt - u_list[at]

                prob_opt_list.append(prob_opt)
                cum_regret_list.append(cum_regret)
            return prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T

    def B_KLUCB(self, arm_list,u_list, K):
        # LB_arm = self.LB_arm
        # UB_arm = self.UB_arm
        # arm_list = self.arm_list
        # u_list = self.u_list

        # arm_list, LB_arm, UB_arm, u_list = self.Arm_Cut()

        ft = lambda x: np.log(x) + 3 * np.log(np.log(x))
        if len(arm_list) == 1:
            print("All cut")
            return arm_list
        else:
            Arm = []
            Na_T = dict()
            sum_reward = 0
            cum_regret = 0
            Reward_arm = dict()

            prob_opt_list = []
            cum_regret_list = []

            # Initial variable setting
            for t in range(len(arm_list)):
                # Before pulling
                at = t
                Na_T[at] = 0
                Reward_arm[at] = []

            # Initial pulling
            for t in range(K*len(arm_list)):
                # Pulling!
                at = np.mod(t, len(arm_list))
                Arm.append(at)
                rt = self.Pull_Receive(at, Na_T)
                Reward_arm[at].append(rt)
                sum_reward += rt
                Na_T[at] += 1
                prob_opt = Na_T[self.opt_arm] / (t + 1)
                cum_regret += self.u_opt - u_list[at]

                prob_opt_list.append(prob_opt)
                cum_regret_list.append(cum_regret)

            # Run!
            UCB_list = []

            # data_collection = {'Arm':Arm, 'Reward':Reward, 'Cum_regret': cum_regret, 'Prob_opt':prob_opt }
            X_hat_list = []
            for t in range(K*len(arm_list), self.T):
                UB_list = []
                for a in arm_list:
                    # standard UCB
                    mu_hat = np.mean(Reward_arm[a])
                    UB_a = self.MaxKL(mu_hat,ft(t),Na_T[a],init_maxval=1)
                    UB_a = np.min([self.UB_arm[a], UB_a])
                    UB_list.append(UB_a)
                UCB_list.append(UB_list)

                at = UB_list.index(max(UB_list))
                Arm.append(at)
                rt = self.Pull_Receive(at, Na_T)

                Reward_arm[at].append(rt)
                sum_reward += rt

                Na_T[at] += 1
                prob_opt = Na_T[self.opt_arm] / (t + 1)
                cum_regret += self.u_opt - u_list[at]

                prob_opt_list.append(prob_opt)
                cum_regret_list.append(cum_regret)
            return prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T

    def Bandit_Run(self):
        prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T = self.KLUCB(self.arm_list,self.u_list,self.K)
        prob_opt_list_B, cum_regret_list_B, UCB_list_B, Arm_B, X_hat_list_B, Na_T_B = self.B_KLUCB(self.arm_list,self.u_list,self.K)
        return [[prob_opt_list, cum_regret_list, UCB_list, Arm, X_hat_list, Na_T],[prob_opt_list_B, cum_regret_list_B, UCB_list_B, Arm_B, X_hat_list_B, Na_T_B]]


