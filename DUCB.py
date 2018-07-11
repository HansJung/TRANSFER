import numpy as np
import scipy as sp
import pandas as pd
import itertools
from scipy.special import lambertw

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

def compute_val(poly_k, poly_j,s, Mkj, eps_t, Xj,Yj,Z ):
    zs = Z.iloc[s]
    xs = Xj.iloc[s]
    ys = Yj.iloc[s]

    poly_ratio_kj = Poly_ratio_kj(poly_k,poly_j,zs,xs)
    return (1/Mkj) * ys * poly_ratio_kj * indicator(poly_ratio_kj, eps_t, Mkj)

def indicator(poly_ratio, eps_t, Mkj):
    ub = 2*np.log(2/eps_t) * Mkj
    if poly_ratio < ub:
        return 1
    else:
        return 0

def Poly_ratio_kj(poly_k, poly_j, zs, xj_s):
    pi_k = poly_k.predict_proba(pd.DataFrame([zs]))[0][xj_s]
    pi_j = poly_j.predict_proba(pd.DataFrame([zs]))[0][xj_s]
    return pi_k / (pi_j+1e-8)

def Upper_bonus(k, Ns, M, policy_idx_list, t):
    Zk_t = 0
    c1 = 16
    for j in policy_idx_list:
        Zk_t += Ns[j] / M[k, j]
    C = (np.sqrt(c1 * t * np.log(t))) / (Zk_t)
    Bt = np.real(C * lambertw(2 / (C + 1e-8)))
    Sk = 1.5 * Bt
    return Sk

