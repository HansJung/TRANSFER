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
        N_obs = len(f_weights)
        N_do = len(g_weights)

        sum_over_i = 0
        for i in range(N_obs):
            Pi_i = f_weights[i]
            sum_over_j = 0
            for j in range(N_obs):
                Pi_j = f_weights[j]
                KL_ij = self.KL_Gaussian(f_means[i], f_stds[i], f_means[j], f_stds[j])
                sum_over_j += Pi_j * np.exp(-KL_ij)

            sum_over_k = 0
            for k in range(N_do):
                tau_k = g_weights[k]
                KL_ik = self.KL_Gaussian(f_means[i], f_stds[i], g_means[k], g_stds[k])
                sum_over_k += tau_k * KL_ik
            sum_over_i += Pi_i * np.log(sum_over_j/sum_over_k)
        return sum_over_i



        # N_obs = len(f_weights)
        # for k in range(N_obs):
        #     reducing = 1e-12
        #     pi_k = f_weights[k]
        #     f_mean_k = f_means[k]
        #     f_stds_k = f_stds[k]
        #
        #     sum_numer = 0
        #     sum_deno = 0
        #     for kdot in range(len(f_weights)):
        #         pi_kdot = f_weights[kdot]
        #         # if pi_kdot <= reducing:
        #         #     continue
        #         f_mean_kdot = f_means[kdot]
        #         f_stds_kdot = f_stds[kdot]
        #
        #         sum_numer += pi_kdot * \
        #                      np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, f_mean_kdot, f_stds_kdot))
        #
        #     for h in range(len(g_weights)):
        #         nu_h = g_weights[h]
        #         # if nu_h <= reducing:
        #         #     continue
        #         g_mean_h = g_means[h]
        #         g_stds_h = g_stds[h]
        #
        #         sum_deno += nu_h * \
        #                     np.exp(-self.KL_Gaussian(f_mean_k, f_stds_k, g_mean_h, g_stds_h))
        #
        #     sum_result += pi_k * np.log( (sum_numer )  / (sum_deno ) )
        # return sum_result

    def deriv_KL_GMM(self, f_weights, g_weights, f_means, g_means, f_stds, g_stds):
        sum_derive = [0] * len(g_weights)
        N_do = len(g_weights)
        N_obs = len(f_weights)

        for l in range(N_do):
            tau_l = g_weights[l]
            g_mean_l = g_means[l]
            g_std_l = g_stds[l]
            sum_over_i = 0
            for i in range(N_obs):
                Pi_i = f_weights[i]
                f_mean_i = f_means[i]
                f_stds_i = f_stds[i]
                numer = tau_l * np.exp(-self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_l, g_std_l))

                sum_over_k = 0
                for k in range(N_do):
                    tau_k = g_weights[k]
                    g_mean_k = g_means[k]
                    g_std_k = g_stds[k]
                    sum_over_k += tau_k * np.exp(-self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_k, g_std_k))

                third_part = (g_mean_l - f_mean_i)/(g_std_l ** 2)
                sum_over_i += Pi_i * (numer/sum_over_k) * third_part
            sum_derive[l] = sum_over_i
                #
                #
                # tau_j = g_weights[j]
                # g_mean_j = g_means[j]
                # g_stds_j = g_stds[j]
                # Aij = self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_j, g_stds_j)
                # exp_Aij = tau_j * np.exp(-Aij)
                #
                # sum_exp_Aijdot = 0
                # for j_dot in range(len(g_weights)):
                #     tau_jdot = g_weights[j_dot]
                #     g_mean_jdot = g_means[j_dot]
                #     g_stds_jdot = g_stds[j_dot]
                #     Aijdot = self.KL_Gaussian(f_mean_i, f_stds_i, g_mean_jdot, g_stds_jdot)
                #     sum_exp_Aijdot += tau_jdot * np.exp(-Aijdot)
                # # sum_derive[j] += (exp_Aij/sum_exp_Aijdot) * ((g_mean_j - f_mean_i) / (g_stds_j ** 2))
                #     # sum_derive[j] += pi_i * exp_Aij / sum_exp_Aijdot * ((g_mean_j - f_mean_i) / (g_stds_j ** 2))

        return sum_derive

    def deriv_KL_GMM_weights(self, f_weights, g_weights, f_means, g_means, f_stds, g_stds):
        sum_derive = [0] * len(g_weights)
        N_do = len(g_weights)
        N_obs = len(f_weights)


        for l in range(N_do):
            tau_l = g_weights[l]
            g_mean_l = g_means[l]
            g_std_l = g_stds[l]

            sum_over_i = 0
            for i in range(N_obs):
                Pi_i = f_weights[i]
                f_mean_i = f_means[i]
                f_std_i = f_stds[i]
                numer = np.exp(-self.KL_Gaussian(f_mean_i, f_std_i, g_mean_l, g_std_l))

                sum_over_k = 0
                for k in range(N_do):
                    tau_k = g_weights[k]
                    g_mean_k = g_means[k]
                    g_std_k = g_stds[k]
                    sum_over_k += tau_k * np.exp(-self.KL_Gaussian(f_mean_i, f_std_i, g_mean_k, g_std_k))
                sum_over_i += Pi_i * (numer/sum_over_k)
            sum_derive[l] = -sum_over_i
        return sum_derive




    def Bound_Optimization_mu(self, C, x0, cls_size, f_means, f_stds, f_weights, g_stds, g_weights, opt_mode):
        cons = ({'type': 'ineq',
                 'fun': lambda x: C - self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds),
                 'jac': lambda x: -1 * self.deriv_KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)},
                {'type': 'ineq',
                 'fun': lambda x: self.KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds),
                 'jac': lambda x: self.deriv_KL_GMM(f_weights, g_weights, f_means, x, f_stds, g_stds)}
                )
        bdds = tuple([tuple([0, 1])] * cls_size)

        min_fun = lambda mu_do: np.sum(mu_do * g_weights)
        min_fun_der = lambda mu_do: g_weights

        max_fun = lambda mu_do: -np.sum(mu_do * g_weights)
        max_fun_der = lambda mu_do: -g_weights

        max_iter = 100
        # x0 = np.random.rand(cls_size)

        if opt_mode == 'max':
            fun = max_fun
            solution = minimize(fun, x0=x0, constraints=cons, jac=max_fun_der, method='Newton-CG', bounds=bdds, tol=1e-12,
                                options={'maxiter': max_iter, 'disp': True})

        elif opt_mode == 'min':
            fun = min_fun
            solution = minimize(fun, x0=x0, constraints=cons, jac=min_fun_der, method='Newton-CG', bounds=bdds, tol=1e-12,
                                options={'maxiter': max_iter, 'disp': True})

        return solution

    def Bound_Optimization_weights(self, C, x0, cls_size, f_means, f_stds, f_weights, g_stds, g_means, opt_mode):

        cons = ({'type': 'ineq',
                 'fun': lambda x: C - self.KL_GMM(f_weights, x, f_means, g_means, f_stds, g_stds),
                 'jac': lambda x: -1 * self.deriv_KL_GMM_weights(f_weights, x, f_means, g_means, f_stds, g_stds)},
                {'type': 'ineq',
                 'fun': lambda x: self.KL_GMM(f_weights, x, f_means, g_means, f_stds, g_stds),
                 'jac': lambda x: self.deriv_KL_GMM_weights(f_weights, x, f_means, g_means, f_stds, g_stds)},
                {'type': 'eq',
                 'fun': lambda x: np.sum(x)-1}
                )
        bdds = tuple([tuple([0, 1])] * cls_size)

        min_fun = lambda w: np.sum(g_means * w)
        min_fun_der = lambda w: g_means

        max_fun = lambda w: -np.sum(g_means * w)
        max_fun_der = lambda w: -1*g_means

        max_iter = 100
        # x0 = np.random.rand(cls_size)

        if opt_mode == 'max':
            fun = max_fun
            solution = minimize(fun, x0=x0, constraints=cons, jac=max_fun_der, method='Newton-CG', bounds=bdds, tol=1e-12,
                                options={'maxiter': max_iter, 'disp': True})

        elif opt_mode == 'min':
            fun = min_fun
            solution = minimize(fun, x0=x0, constraints=cons, jac=min_fun_der, method='Newton-CG', bounds=bdds, tol=1e-12,
                                options={'maxiter': max_iter, 'disp': True})

        return solution

    def Solve_Optimization(self, C, cls_size, iter_opt, opt_mode):
        # Observations
        rounding_digit = 8
        f_means = np.round(self.dpobs.means_, rounding_digit).T[0]
        f_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpobs.covariances_), rounding_digit))
        f_weights = np.round(self.dpobs.weights_, rounding_digit)
        g_stds = f_stds

        prev_mu = copy.copy(f_means)
        prev_weights = copy.copy(f_weights)

        # prev_mu = np.random.random(cls_size)
        # prev_mu = copy.copy(f_means)
        # prev_weights = np.random.dirichlet([1]*cls_size)

        prev_mu_list = [prev_mu]
        prev_weights_list = [prev_weights]

        for iter_idx in range(1,iter_opt):
            curr_mu_solver = self.Bound_Optimization_mu(C,prev_mu_list[iter_idx-1],cls_size,f_means,f_stds,f_weights,g_stds,prev_weights_list[iter_idx-1],opt_mode=opt_mode)
            if opt_mode == 'min':
                prev_mu_list.append(curr_mu_solver.x - (1/(max(20,cls_size)*iter_idx)) *np.random.random(cls_size))
            else:
                prev_mu_list.append(curr_mu_solver.x + (1 / (max(20,cls_size)* iter_idx)) * np.random.random(cls_size))
            curr_weight_solver = self.Bound_Optimization_weights(C,prev_weights_list[iter_idx-1],cls_size,f_means,f_stds,f_weights,g_stds,prev_mu_list[iter_idx],opt_mode=opt_mode)
            prev_weights_list.append(curr_weight_solver.x)

            # Criterion check
            if np.sum(np.abs(prev_mu_list[iter_idx] - prev_mu_list[iter_idx-1])) < 0.001 and np.sum(np.abs(prev_weights_list[iter_idx] - prev_weights_list[iter_idx-1])) < 0.001:
                break

        mu_opt = prev_mu_list[len(prev_mu_list)-1]
        weight_opt = prev_weights_list[len(prev_weights_list)-1]
        opt_mean = np.sum(mu_opt * weight_opt)
        return curr_mu_solver, prev_mu_list, prev_weights_list, opt_mean

    def Conduct_Optimization(self, C, cls_size, iter_opt):
        min_mu_solver, min_mu_list, min_weight_list, LB = self.Solve_Optimization(C, cls_size, iter_opt, 'min')
        max_mu_solver, max_mu_list, max_weight_list, UB = self.Solve_Optimization(C, cls_size, iter_opt, 'max')

        return [LB, UB, min_mu_list,  max_mu_list, min_weight_list, max_weight_list]


        # mu_update




    # rounding_digit = 12
    #
    #     f_means = np.round(self.dpobs.means_, rounding_digit).T[0]
    #     f_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpobs.covariances_), rounding_digit))
    #     f_weights = np.round(self.dpobs.weights_, rounding_digit)
    #     g_stds = f_stds
    #
    #     x0 = np.random.random(cls_size)
    #
    #     ''' Upper '''
    #     # Initial upper variable setting
    #     prev_upper_x0 = copy.copy(x0)
    #     prev_upper_prop = np.array([100] * cls_size)
    #     prev_upper_weight = np.random.dirichlet(prev_upper_prop)
    #     # prev_upper_weight = copy.copy(f_weights)
    #
    #     ## Initial optimization based on initial value
    #     print('initial x0', prev_upper_x0)
    #     prev_upper_mu_solver = self.Bound_Optimization_mu(C, prev_upper_x0, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                          prev_upper_weight, opt_mode='max')
    #     prev_upper_mu = prev_upper_mu_solver.x
    #     prev_upper_weight_solver = self.Bound_Optimization_weights(C, prev_upper_weight, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                                         prev_upper_mu, opt_mode='max')
    #     prev_upper_weight = prev_upper_weight_solver.x
    #
    #     for iter_idx in range(iter_opt):
    #         print(iter_idx, 'th UPPER operation')
    #         # Update based on previous optimization
    #         # prev_upper_rank = rank_compute(prev_upper_mu.x, 'upper')
    #         # curr_upper_prop = prev_upper_prop + np.power(prev_upper_rank, 2)
    #         # curr_upper_weight = np.random.dirichlet(curr_upper_prop)
    #         curr_upper_x0 = copy.copy(prev_upper_mu)
    #         curr_noise = (1/(cls_size*(1+iter_idx))) * np.random.rand(cls_size)
    #         curr_upper_x0 += curr_noise
    #         curr_upper_mu_solver = self.Bound_Optimization_mu(C, curr_upper_x0, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                              prev_upper_weight, opt_mode='max')
    #         curr_upper_mu = curr_upper_mu_solver.x
    #         curr_upper_weight_solver = self.Bound_Optimization_weights(C, prev_upper_weight, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                                                    curr_upper_mu, opt_mode='max')
    #         curr_upper_weight = curr_upper_weight_solver.x
    #
    #         # Compare to previous and current
    #         upper_x_update = np.sum(np.abs(curr_upper_mu - curr_upper_x0))
    #         prev_mean = np.sum(curr_upper_x0 * prev_upper_weight)
    #         current_mean = np.sum(curr_upper_mu * curr_upper_weight)
    #         mean_update = np.abs(current_mean - prev_mean)
    #         weight_update = np.abs(np.sum(prev_upper_weight - curr_upper_weight))
    #
    #         print('Means', prev_mean, current_mean)
    #         print('Weight_update',weight_update)
    #         print('Prev X', curr_upper_x0)
    #         print('Curr X', curr_upper_mu)
    #         print('Curr Prop', curr_upper_weight)
    #
    #         ## Stopping rule
    #         if upper_x_update < cls_size*0.01 or mean_update < 0.01 or weight_update < 0.001:
    #             if upper_x_update < cls_size*0.01:
    #                 print('Sufficient x update!')
    #             elif mean_update < 0.01:
    #                 print('Sufficient mean update!')
    #             elif weight_update < 0.001:
    #                 print('Sufficient weight update')
    #             break
    #
    #         # previous value update
    #         # prev_upper_weight = curr_upper_weight
    #         # prev_upper_prop = copy.copy(curr_upper_prop)
    #         prev_upper_mu = copy.copy(curr_upper_mu)
    #         prev_upper_weight = copy.copy(curr_upper_weight)
    #
    #     ''' Lower '''
    #     # Initial lower variable setting
    #     prev_lower_x0 = copy.copy(x0)
    #     prev_lower_prop = np.array([100] * cls_size)
    #     # prev_lower_weight = copy.copy(f_weights)
    #     prev_lower_weight = np.random.dirichlet(prev_lower_prop)
    #
    #     ## Initial optimization based on initial value
    #     print('initial x0', prev_lower_x0)
    #     prev_lower_mu_solver = self.Bound_Optimization_mu(C, prev_lower_x0, cls_size, f_means, f_stds, f_weights,
    #                                                       g_stds,
    #                                                       prev_lower_weight, opt_mode='min')
    #     prev_lower_mu = prev_lower_mu_solver.x
    #     prev_lower_weight_solver = self.Bound_Optimization_weights(C, prev_lower_weight, cls_size, f_means, f_stds,
    #                                                                f_weights, g_stds,
    #                                                                prev_lower_mu, opt_mode='min')
    #     prev_lower_weight = prev_lower_weight_solver.x
    #
    #     for iter_idx in range(iter_opt):
    #         print(iter_idx, 'th LOWER operation')
    #         # Update based on previous optimization
    #         curr_lower_x0 = copy.copy(prev_lower_mu)
    #         curr_noise = (1 / (cls_size * (1 + iter_idx))) * np.random.rand(cls_size)
    #         curr_lower_x0 += curr_noise
    #         curr_lower_mu_solver = self.Bound_Optimization_mu(C, curr_lower_x0, cls_size, f_means, f_stds, f_weights,
    #                                                           g_stds,
    #                                                           prev_lower_weight, opt_mode='min')
    #         curr_lower_mu = curr_lower_mu_solver.x
    #         curr_lower_weight_solver = self.Bound_Optimization_weights(C, prev_lower_weight, cls_size, f_means, f_stds,
    #                                                                    f_weights, g_stds,
    #                                                                    curr_lower_mu, opt_mode='min')
    #         curr_lower_weight = curr_lower_weight_solver.x
    #
    #         # Compare to previous and current
    #         lower_x_update = np.sum(np.abs(curr_lower_mu - curr_lower_x0))
    #         prev_mean = np.sum(curr_lower_x0 * prev_lower_weight)
    #         current_mean = np.sum(curr_lower_mu * curr_lower_weight)
    #         mean_update = np.abs(current_mean - prev_mean)
    #         weight_update = np.abs(np.sum(prev_lower_weight - curr_lower_weight))
    #
    #         print('Means', prev_mean, current_mean)
    #         print('Weight_update', weight_update)
    #         print('Prev X', curr_lower_x0)
    #         print('Curr X', curr_lower_mu)
    #         print('Curr Prop', curr_lower_weight)
    #
    #         ## Stopping rule
    #         if lower_x_update < cls_size * 0.01 or mean_update < 0.01 or weight_update < 0.001:
    #             if lower_x_update < cls_size * 0.01:
    #                 print('Sufficient x update!')
    #             elif mean_update < 0.01:
    #                 print('Sufficient mean update!')
    #             elif weight_update < 0.001:
    #                 print('Sufficient weight update')
    #             break
    #
    #         prev_lower_mu = copy.copy(curr_lower_mu)
    #         prev_lower_weight = copy.copy(curr_lower_weight)
    #
    #     LB = np.sum(curr_lower_mu * curr_lower_weight)
    #     UB = np.sum(curr_upper_mu * curr_upper_weight)
    #
    #     return [LB, UB, curr_lower_mu_solver, curr_upper_mu_solver, curr_lower_weight, curr_upper_weight]

    # def Solve_Optimization(self, C, cls_size, iter_opt):
    #     rounding_digit = 12
    #
    #     f_means = np.round(self.dpobs.means_, rounding_digit).T[0]
    #     f_stds = np.ndarray.flatten(np.round(np.sqrt(self.dpobs.covariances_), rounding_digit))
    #     f_weights = np.round(self.dpobs.weights_, rounding_digit)
    #     g_stds = f_stds
    #
    #     x0 = np.random.random(cls_size)
    #
    #     ''' Upper '''
    #     # Initial upper variable setting
    #     prev_upper_x0 = copy.copy(x0)
    #     prev_upper_prop = np.array([100] * cls_size)
    #     prev_upper_weight = np.random.dirichlet(prev_upper_prop)
    #     # prev_upper_weight = copy.copy(f_weights)
    #
    #     ## Initial optimization based on initial value
    #     print('initial x0', prev_upper_x0)
    #     prev_upper_mu_solver = self.Bound_Optimization_mu(C, prev_upper_x0, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                          prev_upper_weight, opt_mode='max')
    #     prev_upper_mu = prev_upper_mu_solver.x
    #     prev_upper_weight_solver = self.Bound_Optimization_weights(C, prev_upper_weight, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                                         prev_upper_mu, opt_mode='max')
    #     prev_upper_weight = prev_upper_weight_solver.x
    #
    #     for iter_idx in range(iter_opt):
    #         print(iter_idx, 'th UPPER operation')
    #         # Update based on previous optimization
    #         # prev_upper_rank = rank_compute(prev_upper_mu.x, 'upper')
    #         # curr_upper_prop = prev_upper_prop + np.power(prev_upper_rank, 2)
    #         # curr_upper_weight = np.random.dirichlet(curr_upper_prop)
    #         curr_upper_x0 = copy.copy(prev_upper_mu)
    #         curr_noise = (1/(cls_size*(1+iter_idx))) * np.random.rand(cls_size)
    #         curr_upper_x0 += curr_noise
    #         curr_upper_mu_solver = self.Bound_Optimization_mu(C, curr_upper_x0, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                              prev_upper_weight, opt_mode='max')
    #         curr_upper_mu = curr_upper_mu_solver.x
    #         curr_upper_weight_solver = self.Bound_Optimization_weights(C, prev_upper_weight, cls_size, f_means, f_stds, f_weights, g_stds,
    #                                                                    curr_upper_mu, opt_mode='max')
    #         curr_upper_weight = curr_upper_weight_solver.x
    #
    #         # Compare to previous and current
    #         upper_x_update = np.sum(np.abs(curr_upper_mu - curr_upper_x0))
    #         prev_mean = np.sum(curr_upper_x0 * prev_upper_weight)
    #         current_mean = np.sum(curr_upper_mu * curr_upper_weight)
    #         mean_update = np.abs(current_mean - prev_mean)
    #         weight_update = np.abs(np.sum(prev_upper_weight - curr_upper_weight))
    #
    #         print('Means', prev_mean, current_mean)
    #         print('Weight_update',weight_update)
    #         print('Prev X', curr_upper_x0)
    #         print('Curr X', curr_upper_mu)
    #         print('Curr Prop', curr_upper_weight)
    #
    #         ## Stopping rule
    #         if upper_x_update < cls_size*0.01 or mean_update < 0.01 or weight_update < 0.001:
    #             if upper_x_update < cls_size*0.01:
    #                 print('Sufficient x update!')
    #             elif mean_update < 0.01:
    #                 print('Sufficient mean update!')
    #             elif weight_update < 0.001:
    #                 print('Sufficient weight update')
    #             break
    #
    #         # previous value update
    #         # prev_upper_weight = curr_upper_weight
    #         # prev_upper_prop = copy.copy(curr_upper_prop)
    #         prev_upper_mu = copy.copy(curr_upper_mu)
    #         prev_upper_weight = copy.copy(curr_upper_weight)
    #
    #     ''' Lower '''
    #     # Initial lower variable setting
    #     prev_lower_x0 = copy.copy(x0)
    #     prev_lower_prop = np.array([100] * cls_size)
    #     # prev_lower_weight = copy.copy(f_weights)
    #     prev_lower_weight = np.random.dirichlet(prev_lower_prop)
    #
    #     ## Initial optimization based on initial value
    #     print('initial x0', prev_lower_x0)
    #     prev_lower_mu_solver = self.Bound_Optimization_mu(C, prev_lower_x0, cls_size, f_means, f_stds, f_weights,
    #                                                       g_stds,
    #                                                       prev_lower_weight, opt_mode='min')
    #     prev_lower_mu = prev_lower_mu_solver.x
    #     prev_lower_weight_solver = self.Bound_Optimization_weights(C, prev_lower_weight, cls_size, f_means, f_stds,
    #                                                                f_weights, g_stds,
    #                                                                prev_lower_mu, opt_mode='min')
    #     prev_lower_weight = prev_lower_weight_solver.x
    #
    #     for iter_idx in range(iter_opt):
    #         print(iter_idx, 'th LOWER operation')
    #         # Update based on previous optimization
    #         curr_lower_x0 = copy.copy(prev_lower_mu)
    #         curr_noise = (1 / (cls_size * (1 + iter_idx))) * np.random.rand(cls_size)
    #         curr_lower_x0 += curr_noise
    #         curr_lower_mu_solver = self.Bound_Optimization_mu(C, curr_lower_x0, cls_size, f_means, f_stds, f_weights,
    #                                                           g_stds,
    #                                                           prev_lower_weight, opt_mode='min')
    #         curr_lower_mu = curr_lower_mu_solver.x
    #         curr_lower_weight_solver = self.Bound_Optimization_weights(C, prev_lower_weight, cls_size, f_means, f_stds,
    #                                                                    f_weights, g_stds,
    #                                                                    curr_lower_mu, opt_mode='min')
    #         curr_lower_weight = curr_lower_weight_solver.x
    #
    #         # Compare to previous and current
    #         lower_x_update = np.sum(np.abs(curr_lower_mu - curr_lower_x0))
    #         prev_mean = np.sum(curr_lower_x0 * prev_lower_weight)
    #         current_mean = np.sum(curr_lower_mu * curr_lower_weight)
    #         mean_update = np.abs(current_mean - prev_mean)
    #         weight_update = np.abs(np.sum(prev_lower_weight - curr_lower_weight))
    #
    #         print('Means', prev_mean, current_mean)
    #         print('Weight_update', weight_update)
    #         print('Prev X', curr_lower_x0)
    #         print('Curr X', curr_lower_mu)
    #         print('Curr Prop', curr_lower_weight)
    #
    #         ## Stopping rule
    #         if lower_x_update < cls_size * 0.01 or mean_update < 0.01 or weight_update < 0.001:
    #             if lower_x_update < cls_size * 0.01:
    #                 print('Sufficient x update!')
    #             elif mean_update < 0.01:
    #                 print('Sufficient mean update!')
    #             elif weight_update < 0.001:
    #                 print('Sufficient weight update')
    #             break
    #
    #         prev_lower_mu = copy.copy(curr_lower_mu)
    #         prev_lower_weight = copy.copy(curr_lower_weight)
    #
    #     LB = np.sum(curr_lower_mu * curr_lower_weight)
    #     UB = np.sum(curr_upper_mu * curr_upper_weight)
    #
    #     return [LB, UB, curr_lower_mu_solver, curr_upper_mu_solver, curr_lower_weight, curr_upper_weight]