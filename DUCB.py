import numpy as np
import pandas as pd
import itertools

# DUCB function
## Compute M
class DUCB(object):
    def __init__(self,policy_list,pred_list, opt_pl, T, X_pl_list,Y_pl_list,Z):
        Mmat = self.Matrix_M(policy_list,X_pl_list,Z)
        self.param_set = [policy_list, pred_list, Mmat, X_pl_list, Y_pl_list, Z]
        self.opt_pl = opt_pl
        self.T = T

    def Compute_divergence_two(self, poly_k, poly_j, Z, Xk):
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

    ## Compute val
    def indicator(self, poly_ratio, eps_t, Mkj):
        ub = 2*np.log(2/eps_t) * Mkj
        if poly_ratio < ub:
            return 1
        else:
            return 0

    def est_predict_proba(self,poly, zs, xj_s):
        try:
            result = poly.predict_proba(zs)[0][xj_s]
        except:
            zs = pd.DataFrame(np.matrix(zs))
            result = poly.predict_proba(zs)[0][xj_s]
        return result


    def Poly_ratio_kj(self, poly_k, poly_j, zs, xj_s):
        pi_k = self.est_predict_proba(poly_k,zs,xj_s)
        pi_j = self.est_predict_proba(poly_j,zs,xj_s)
        return pi_k / (pi_j+1e-8)

    def compute_val(self,k, j, s, eps_t, param_set):
        policy_list, pred_list, Mmat, X_pl_list, Y_pl_list, Z = param_set
        poly_k = policy_list[k]
        poly_j = policy_list[j]
        Mkj = Mmat[k,j]

        Xj = X_pl_list[j]
        Yj = Y_pl_list[j]

        zs = Z.iloc[s]
        xjs = int(Xj[s])
        yjs = Yj.iloc[s]

        poly_ratio_kj = self.Poly_ratio_kj(poly_k,poly_j,zs,xjs)
        return (1/Mkj) * yjs * poly_ratio_kj * self.indicator(poly_ratio_kj, eps_t, Mkj)

    # Compute upper bonus
    def upper_bonus(self,  t, k, G, c1 = 16):
        Gk = G[k]
        return np.sqrt(c1*t*np.log(t)) / Gk

    def conduct_DUCB(self):
        policy_list, pred_list, Mmat, X_pl_list, Y_pl_list, Z = self.param_set

        dictV = dict()
        sumV = dict()
        G = dict()
        u = dict()
        num_pull = dict()
        arm_stored = list()
        prob_opt_list = list()
        avg_loss_list = list()
        acc_loss = 0

        for k in range(len(policy_list)):
            dictV[k] = [0] * len(policy_list)
            sumV[k] = 0
            G[k] = 0
            u[k] = 0
            num_pull[k] = 0

        # Initial running
        s = 1
        for k in range(len(policy_list)):
            for j in range(len(policy_list)):
                dictV[k][j] = self.compute_val(k, j, s, 1, self.param_set)
                sumV[k] += dictV[k][j]
                G[k] += 1 / Mmat[k, j]
            sumV[k] = sumV[k] / G[k]
            sk = self.upper_bonus(1, k, G)
            u[k] = sumV[k] + 1.5 * sk

        a_prev = np.argmax(list(u.values()))
        arm_stored.append(a_prev)
        num_pull[a_prev] += 1

        for t in range(len(policy_list) + 1, self.T):
            eps_t = 2 / t
            for k in range(len(policy_list)):
                for j in range(len(policy_list)):
                    Mkj = Mmat[k, j]
                    if j == a_prev:
                        dictV[k][j] += self.compute_val(k, j, t - 1, eps_t, self.param_set)
                        G[k] += 1 / Mkj
                    sumV[k] += dictV[k][j]
                sk = self.upper_bonus(t, k, G)
                sumV[k] = sumV[k] / G[k]
                u[k] = sumV[k] + 1.5 * sk
            a_prev = np.argmax(list(u.values()))
            arm_stored.append(a_prev)
            num_pull[a_prev] += 1
            prob_opt = num_pull[self.opt_pl] / t
            acc_loss += 1 - Y_pl_list[a_prev].iloc[t]
            avg_loss = acc_loss / t
            prob_opt_list.append(prob_opt)
            avg_loss_list.append(avg_loss)
        return [prob_opt_list,avg_loss_list,num_pull]

        # plt.figure()
        # plt.plot(prob_opt_list)
        # plt.figure()
        # plt.plot(avg_loss_list)

