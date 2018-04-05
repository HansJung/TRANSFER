import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from cvxpy import *



class CausalBound(object):
    def __init__(self, dpobs, C):
        '''

        :param dpobs: DPMM encoded dataset
        :param C: Upper constant over KL
        '''
        self.dpobs = dpobs
        self.C = C

    def KL_Gaussian(self, f_mean, f_std, g_mean, g_std):  # KL(f || g)
        return np.log(g_std / f_std) + \
               (f_std ** 2 + (f_mean - g_mean) ** 2) / (2 * g_std ** 2) - \
               0.5

    def Easy_bound(self, f_mean, f_std, g_std, C):
        def M(f_std, g_std, C):
            return 2*(g_std**2)*(C + 0.5 - np.log(g_std) + np.log(f_std) - ((f_std ** 2)/(2*(g_std ** 2))))

        UB = f_mean + np.sqrt(M(f_std, g_std, C))
        LB = f_mean - np.sqrt(M(f_std, g_std, C))
        return [LB,UB]

    def Easy_opt(self, C, f_std, g_std):
        # obs_covs = np.reshape(self.dpobs.covariances_, (1, len(self.dpobs.covariances_)))[0]
        # obs_stds = np.sqrt(obs_covs)
        # obs_std = np.sum(self.dpobs.weights_ * obs_covs)
        # f_std = copy.copy(obs_std)
        # g_std = copy.copy(obs_std)

        f_mean = np.sum((self.dpobs.weights_ * self.dpobs.means_.T)[0])
        return self.Easy_bound(f_mean,f_std,g_std, C)








    def KL_GMM(self, f_weights, g_weights, f_means, g_means, f_stds, g_stds):
        sum_result = 0
        for k in range(len(f_weights)):
            reducing = 1e-12
            pi_k = f_weights[k]
            f_mean_k = f_means[k]
            f_stds_k = f_stds[k]
            # if pi_k <= reducing:
            #     continue

            sum_numer = 0
            sum_deno = 0
            for kdot in range(len(f_weights)):
                pi_kdot = f_weights[kdot]
                # if pi_kdot <= reducing:
                #     continue
                f_mean_kdot = f_means[kdot]
                f_stds_kdot = f_stds[kdot]

                sum_numer += pi_kdot * \
                             np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, f_mean_kdot, f_stds_kdot))

            for h in range(len(g_weights)):
                nu_h = g_weights[h]
                # if nu_h <= reducing:
                #     continue
                g_mean_h = g_means[h]
                g_stds_h = g_stds[h]

                sum_deno += nu_h * \
                            np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, g_mean_h, g_stds_h))

            sum_result += pi_k * np.log( (sum_numer + reducing)  / (sum_deno + reducing) )
        return sum_result

    def deriv_KL_GMM(self, f_weights, g_weights, f_means, g_means, f_stds, g_stds, mode):
        sum_derive = [0] * len(g_weights)

        for j in range(len(g_weights)):
            for i in range(len(f_weights)):
                pi_i = f_weights[i]
                f_mean_i = f_means[i]
                f_stds_i = f_stds[i]

                tau_j = g_weights[j]
                g_mean_j = g_means[j]
                g_stds_j = g_stds[j]
                Aij = self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_j, g_stds_j)
                exp_Aij = tau_j * np.exp(-Aij)

                sum_exp_Aijdot = 0
                for j_dot in range(len(g_weights)):
                    tau_jdot = g_weights[j_dot]
                    g_mean_jdot = g_means[j_dot]
                    g_stds_jdot = g_stds[j_dot]
                    Aijdot = self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_jdot, g_stds_jdot)
                    sum_exp_Aijdot += tau_jdot * np.exp(-Aijdot)
                if mode == 'min':
                    sum_derive[j] += pi_i * exp_Aij / sum_exp_Aijdot * ((g_mean_j - f_mean_i) / (g_stds_j ** 2))
                elif mode == 'max':
                    sum_derive[j] += - (pi_i * exp_Aij / sum_exp_Aijdot * ((g_mean_j - f_mean_i) / (g_stds_j ** 2)))
        return sum_derive

    def Solve_Optimization(self, C, cls_size, iter_opt):
        rounding_digit = 12

        f_means = np.round(self.dpobs.means_, rounding_digit)
        f_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpobs.covariances_), rounding_digit))
        f_weights = np.round(self.dpobs.weights_, rounding_digit)

        g_stds = np.array([1 / cls_size] * cls_size)

        # upper_update = np.array([1]*cls_size) + self.dpobs.weights_.argsort().argsort()
        # upper_weight = np.random.dirichlet(upper_update)
        upper_update = np.array([1]*cls_size)
        upper_weight = np.mean(np.random.dirichlet(upper_update,100),axis=0)

        # lower_update = np.array([1] * cls_size) + (max(self.dpobs.weights_.argsort().argsort()) - self.dpobs.weights_.argsort().argsort())
        # lower_weight = np.random.dirichlet(lower_update)
        lower_update = np.array([1]*cls_size)
        lower_weight = np.mean(np.random.dirichlet(lower_update,100),axis=0)

        # x0 = np.random.rand(cls_size)
        x0 = f_means

        print(0, " th iteration")
        upper = self.Bound_Optimization(C, x0, cls_size, f_means, f_stds, f_weights, g_stds, upper_weight, opt_mode='max')
        lower = self.Bound_Optimization(C, x0, cls_size, f_means, f_stds, f_weights, g_stds, lower_weight, opt_mode='min')

        for iter_idx in range(iter_opt):
            ### Update
            prev_upper_weight = upper_weight
            print("-"*50)
            print(iter_idx+1," th iteration")
            upper_rank = upper.x.argsort().argsort()
            upper_update += upper_rank
            upper_weight = np.mean(np.random.dirichlet(upper_update, 100),axis=0)
            upper_update_quantity = np.sum(np.abs(upper_weight - prev_upper_weight))
            print('upper_weight', upper_weight)
            print('upper update', upper_update_quantity)
            upper_starting = upper.x + (1/(10*(1+iter_idx))) * np.random.rand(cls_size)
            upper = self.Bound_Optimization(C, upper_starting, cls_size, f_means, f_stds, f_weights, g_stds, upper_weight, opt_mode='max')

            prev_lower_weight = lower_weight
            lower_rank = lower.x.argsort().argsort()
            lower_rank = max(lower_rank) - lower_rank
            lower_update += lower_rank
            lower_weight = np.mean(np.random.dirichlet(lower_update, 100),axis=0)
            lower_update_quantity = np.sum(np.abs(lower_weight - prev_lower_weight))
            print('lower_weight', lower_weight)
            print('lower update', lower_update_quantity)
            lower_starting = lower.x+ (1/(10*(1+iter_idx))) * np.random.rand(cls_size)
            lower = self.Bound_Optimization(C, lower_starting, cls_size, f_means, f_stds, f_weights, g_stds, lower_weight, opt_mode='min')
            print("-"*50)

            if max(upper_update_quantity, lower_update_quantity) < 0.01:
                print('Sufficient update!')
                break

        LB = np.sum(lower.x * lower_weight)
        UB = np.sum(upper.x * upper_weight)

        return [LB, UB, lower, upper]


    def Bound_Optimization(self, C, x0, cls_size, f_means, f_stds, f_weights, g_stds, g_weights, opt_mode):

        cons = ({'type': 'ineq',
                 'fun': lambda x: np.array(C - self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)[0])},
                {'type': 'ineq',
                 'fun': lambda x: np.array(self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)[0])}
                )
        bdds = tuple([tuple([0, 1])] * cls_size)

        min_fun = lambda mu_do: np.sum(mu_do * g_weights)
        max_fun = lambda mu_do: -np.sum(mu_do * g_weights)

        max_iter = 100
        # x0 = np.random.rand(cls_size)

        if opt_mode == 'max':
            fun = max_fun
            jac_derive = lambda mu_do: self.deriv_KL_GMM(f_weights, g_weights, f_means, mu_do, f_stds, g_stds,mode='max')
            solution = minimize(fun, x0=x0, constraints=cons, jac=jac_derive, method='TNC', bounds=bdds, tol=1e-8,
                             options={'maxiter': max_iter, 'disp': True})

        elif opt_mode == 'min':
            fun = min_fun
            jac_derive = lambda mu_do: self.deriv_KL_GMM(f_weights, g_weights, f_means, mu_do, f_stds, g_stds,
                                                         mode='min')
            solution = minimize(fun, x0=x0, constraints=cons, jac=jac_derive, method='TNC', bounds=bdds, tol=1e-8,
                                options={'maxiter': max_iter, 'disp': True})

        return solution


        # self.LB = np.sum(lower.x * g_weights)
        # self.UB = np.sum(upper.x * g_weights)

        # return [self.LB, self.UB, lower, upper]