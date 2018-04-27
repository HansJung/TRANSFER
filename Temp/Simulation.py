import numpy as np
import scipy as sp
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy
import pickle

def KL_Gaussian(f_mean, f_std, g_mean, g_std):  # KL(f || g)
    return np.log(g_std / f_std) + \
           (f_std ** 2 + (f_mean - g_mean) ** 2) / (2 * g_std ** 2) - \
           0.5

def KL_GMM(f_weights, g_weights, f_means, g_means, f_stds, g_stds):
    N_obs = len(f_weights)
    N_do = len(g_weights)

    sum_over_i = 0
    for i in range(N_obs):
        Pi_i = f_weights[i]
        sum_over_j = 0
        for j in range(N_obs):
            Pi_j = f_weights[j]
            KL_ij = KL_Gaussian(f_means[i], f_stds[i], f_means[j], f_stds[j])
            sum_over_j += Pi_j * np.exp(-KL_ij)

        sum_over_k = 0
        for k in range(N_do):
            tau_k = g_weights[k]
            KL_ik = KL_Gaussian(f_means[i], f_stds[i], g_means[k], g_stds[k])
            sum_over_k += tau_k * np.exp(KL_ik)
        sum_over_i += Pi_i * np.log(sum_over_j / (sum_over_k+1e-8))
    return sum_over_i

def GMM_pdf(x, f_weights, f_means, f_stds):
    n_mixture = len(f_weights)
    pdf_sum = 0
    for idx in range(n_mixture):
        pdf_sum += f_weights[idx] * norm.pdf(x,f_means[idx],f_stds[idx])
    return pdf_sum


def KL_GMM_MCMC(f_weights, g_weights, f_means, g_means, f_stds, g_stds, n_samples):
    sample_f = []
    cls_f = []

    for idx in range(n_samples):
        chosen_idx = np.random.choice(list(range(len(f_weights))), size=1, p=f_weights)[0]
        cls_f.append(chosen_idx)
        mu_idx = f_means[chosen_idx]
        sig_idx = f_stds[chosen_idx]
        sample_idx = np.random.normal(mu_idx, sig_idx)
        sample_f.append(sample_idx)

    log_sum = 0
    for idx in range(n_samples):
        fi = sample_f[idx]
        pdf_fi = GMM_pdf(fi,f_weights,f_means,f_stds)
        pdf_gi = GMM_pdf(fi, g_weights, g_means, g_stds)
        log_sum += np.log(pdf_fi/(pdf_gi + 1e-8))
    return np.mean(log_sum)



def data_gen(n_mixture, n_samples,seednum):
    np.random.seed(seednum)
    pi = np.random.dirichlet([10] * n_mixture)
    mu = np.random.choice(np.random.random(n_samples),n_mixture)
    # while max(mu) > 0.7:
    #     mu = np.random.choice(np.random.random(n_samples), n_mixture)

    sig = np.random.gamma(0.5, 0.5, n_mixture)
    param = dict()
    param['mu'] = mu
    param['sig'] = sig
    param['pi'] = pi

    samples = []
    for idx in range(n_samples):
        chosen_idx = np.random.choice(list(range(n_mixture)), size=1, p=pi)[0]
        mu_idx = mu[chosen_idx]
        sig_idx = sig[chosen_idx]
        sample_idx = np.random.normal(mu_idx, sig_idx)
        samples.append(sample_idx)
    samples = np.array(samples)
    # samples = (samples - np.min(samples))/(max(samples) - min(samples))
    return samples, param

def param_do_gen(param_obs,C,n_param):
    params = dict()
    params['mu'] = []
    params['pi'] = []
    params['sig'] = []
    params['max_mu'] = []
    params['average_mu'] = []
    params['KL'] = []
    params['self_KL'] = []
    params['True_KL'] = []

    iter_idx = 0

    while len(params['mu']) <= n_param:
        n_mixture = np.random.randint(low=1, high=10)
        # mu = np.random.choice(np.random.random(100), n_mixture)
        mu = []
        for idx in range(n_mixture):
            chosen_mu = np.random.choice(param_obs['mu'],size=1,p=param_obs['pi'])[0]
            mu.append(chosen_mu)
        if np.random.rand() > 0.5:
            mu = np.array(mu) + np.random.rand()*np.random.rand() * np.random.random(n_mixture)
        else:
            mu = np.array(mu) - np.random.rand()*np.random.rand() * np.random.random(n_mixture)

        pi = np.random.dirichlet([10] * n_mixture)
        sig = np.random.gamma(0.5, 0.5, n_mixture)

        KL_val = KL_GMM_MCMC(param_obs['pi'], pi, param_obs['mu'], mu, param_obs['sig'], sig, 100)
        KL_val_default = 0

        # KL_val = KL_GMM(param_obs['pi'], pi, param_obs['mu'], mu, param_obs['sig'], sig)
        # KL_val_default = KL_GMM(pi, pi, mu, mu, sig, sig)
        # KL_val_default = 0
        if KL_val < C and KL_val > KL_val_default:
        # if KL_val < C and KL_val >= KL_val_default:
            params['mu'].append(mu)
            params['pi'].append(pi)
            params['sig'].append(sig)
            params['max_mu'].append(max(mu))
            params['average_mu'].append(sum(mu*pi))
            params['KL'].append(KL_val)
            params['self_KL'].append(KL_val_default)
            params['True_KL'].append(KL_GMM_MCMC(param_obs['pi'], pi, param_obs['mu'], mu, param_obs['sig'], sig, 100))

        iter_idx += 1
        print('do', iter_idx, len(params['mu']))
        if len(params['mu']) > 100:
            break
    return params

def param_obs_gen(param_obs,C, n_param):
    n_mixture = len(param_obs['pi'])
    sig = copy.copy(param_obs['sig'])
    params = dict()
    params['mu'] = []
    params['pi'] = []
    params['sig'] = []
    params['max_mu'] = []
    params['average_mu'] = []
    params['KL'] = []
    params['True_KL'] = []
    params['self_KL'] = []
    param_obs_mu = copy.copy(param_obs['mu'])

    iter_idx = 0

    while len(params['mu']) <= n_param:
        # mu = np.random.choice(np.random.random(100), n_mixture)
        if np.random.rand() > 0.5:
            mu = np.array(param_obs_mu) + np.random.rand()*np.random.rand()*np.array(np.random.random(n_mixture))
        else:
            mu = np.array(param_obs_mu) - np.random.rand()*np.random.rand() * np.array(np.random.random(n_mixture))
        pi = np.random.dirichlet([10] * n_mixture)

        KL_val = KL_GMM_MCMC(copy.copy(param_obs['pi']),pi, param_obs['mu'], mu, sig, sig, 100)
        KL_val_default = 0

        # KL_val = KL_GMM(param_obs['pi'], pi, param_obs['mu'], mu, sig, sig)
        # KL_val_default = KL_GMM(pi, pi, mu, mu, sig, sig)
        # KL_val_default = 0
        if KL_val < C and KL_val >= KL_val_default:
        # if KL_val < C - KL_val_default:
            params['mu'].append(mu)
            params['pi'].append(pi)
            params['sig'].append(sig)
            params['max_mu'].append(max(mu))
            params['average_mu'].append(sum(mu*pi))
            params['KL'].append(KL_val)
            params['self_KL'].append(KL_val_default)
            params['True_KL'].append(KL_GMM_MCMC(param_obs['pi'],pi, param_obs['mu'], mu, sig, sig, 100))

        iter_idx += 1
        print('obs', iter_idx, len(params['mu']))
        if len(params['mu']) > 100:
            break
    return params


n_obs = 2
n_samples = 5000
seed_obs = np.random.randint(10000000)
# seed_obs = 12345

C = 10
print(seed_obs)

OBS, param_obs = data_gen(n_obs, n_samples,seed_obs)
param_obshat = param_obs_gen(param_obs,C,n_param = 100)
param_do = param_do_gen(param_obs,C,n_param = 100)

print("")
print('observed mu', max(param_obs['mu']))
print('obshat max',max(param_obshat['max_mu']))
print('average_mu',max(param_do['average_mu']))
print('do max',max(param_do['max_mu']))
print("")
# print(sum(1 for x in range(len(param_obshat['max_mu'])) if  param_do['average_mu'][x] > param_obshat['max_mu'][x]))
# print(sum(1 for x in range(len(param_obshat['max_mu'])) if  param_do['average_mu'][x] > param_obshat['max_mu'][x]))
print(max(param_do['average_mu']) < max(param_obshat['max_mu']))
print("")
print('obshat average KL', np.mean(param_obshat['KL']), np.var(param_obshat['KL']))
print('do average KL', np.mean(param_do['KL']), np.var(param_do['KL']))



# plt.figure()
# plt.hist(param_obshat['max_mu'],10)
#
# plt.figure()
# plt.hist(param_do['max_mu'],10)

pd.DataFrame(param_obshat['max_mu']).plot(kind='density')
pd.DataFrame(param_do['max_mu']).plot(kind='density')
# pd.DataFrame(param_do['average_mu']).plot(kind='density')

# pickle.dump([param_obs, param_obshat, param_do],open('Temp/' + str(seed_obs) + '_' + str(C)+ '_' +  str(n_obs) + '.pkl','wb'))