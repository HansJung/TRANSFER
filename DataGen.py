import numpy as np
import pandas as pd
import scipy.special as sp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


class DataGen(object):
    '''
    Class DataGen contains all data and methods related to
    generating data.
    - Input: D (dim), N (num_obs), Ns (num_sample_intv), Mode (easy, crazy)
    - Output: Observational data and Interventional data
    '''

    # Seed fix
    # Some good seed are 1234, 12345, 123456
    # np.random.seed(180406)
    # np.random.seed(160702)
    # np.random.seed(160728)

    def __init__(self,D,N,seed_num):
        np.random.seed(seed_num)
        '''
        Initializing the class
        :param D: Dimension
        :param N: Number of observations and interventions (true)
        :param Ns: Number of sampled observation, Ns << N.
        :param Mode:
            - Easy: easy parametrization (Normal), assuming SCM is known
            - Crazy: crazy parametrization, arbitrary density
        '''

        '''
        Declaring input as global in the class
        to contain the input value as data. 
        '''
        self.dim = D
        self.num_obs = N

    def intfy(self,W):
        '''
        Transform the real-number array (e.g., 1.0,2.0) to
        the integer array (e.g., 1, 2).
        :param W: Real-number array
        :return: Integer array
        '''
        return np.array(list(map(int, W)))

    def inverse_logit(self,Z):
        return np.exp(Z) / (np.exp(Z) + 1)

    def weird_projection(self, X):
        '''
        Project
        :param X:
        :return:
        '''
        X_proj = np.log(np.abs(X)+10) * np.arctan(X) + (10*np.sin(X)) - (np.tan(X ** 2) ) * np.exp(np.power(np.abs(X),0.5))
        X_proj = normalize(X_proj)

        # X_proj = np.log(np.abs(X) + 20) * np.arctan(X ** 2) + np.exp(np.sin(X)) - np.log(np.tanh(X) ** 2 + 2) * np.exp(X)
        # X_proj = (X_proj - np.mean(X_proj, axis=0)) / np.var(X_proj)
        return X_proj

    def gen_U(self):
        mu1 = np.random.rand(self.dim)
        mu2 = np.random.rand(self.dim)
        mu3 = np.random.rand(self.dim)
        cov1 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())
        cov2 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())
        cov3 = np.dot(np.random.rand(self.dim, self.dim),
                      np.random.rand(self.dim, self.dim).transpose())

        U1 = np.random.multivariate_normal(mu1, cov1, self.num_obs)
        U2 = np.random.multivariate_normal(mu2, cov2, self.num_obs)
        U3 = np.random.multivariate_normal(mu3, cov3, self.num_obs)

        self.U1 = U1
        self.U2 = U2
        self.U3 = U3

    def fZ(self, U1, U2):
        Z = np.exp(U1) - np.exp(U2) + U1 - U2
        Z = sp.expit(Z)
        # Z = self.intfy(Z)
        Z = np.reshape(Z,(len(Z),self.dim))
        return Z

    def fX(self, U1, U3, Z):
        coef_xz = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u1x = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3x = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))

        X = sp.expit(np.dot(Z,coef_xz) - 2*np.dot(U3,coef_u3x) + 1*np.exp(np.dot(U1,coef_u1x)) + np.mean(U1) - np.mean(U3) )
        X = np.reshape(X,(len(X),1))
        X = np.round(X)
        return X

    def fY(self, U2, U3, X, Z):
        coef_zy = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u2y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))

        part1 = -2*np.sin(np.dot(U2, coef_u2y))
        part2 = np.power(np.dot(U3,coef_u3y),1)
        part3 = 2*np.array(X)-1
        part4 = np.power(np.abs(np.dot(Z,coef_zy)),2)

        Y = part1 + part2 + part3 + part4
        Y = sp.expit(Y)
        return Y


    def obs_data(self):
        self.gen_U()
        U1 = self.U1
        U2 = self.U2
        U3 = self.U3

        Z = self.fZ(U1, U2)
        X = self.fX(U1,U3,Z)
        Y = self.fY(U2,U3,X,Z)

        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)

        Obs_X = pd.DataFrame(X_obs)
        Obs_Y = pd.DataFrame(Y_obs)
        Obs_Z = pd.DataFrame(Z_obs)

        Obs_XY = pd.concat([Obs_X, Obs_Y], axis=1)
        Obs_XY.columns = ['X', 'Y']
        Obs_Z.columns = range(self.dim)

        Obs = pd.concat([Obs_XY, Obs_Z], axis=1)
        return Obs

    def poly_intv_data(self, pl, Z):
        U1 = self.U1
        U2 = self.U2
        U3 = self.U3

        pl_proba = pl.predict_proba(Z)
        Xi = []
        for idx in range(len(pl_proba)):
            prob_elem = pl_proba[idx]
            prob_one = prob_elem[1]
            xi = round(np.mean(np.random.binomial(1, prob_one, 10)))
            Xi.append(xi)
        X = np.reshape(Xi, (len(Xi), 1))
        Y = self.fY(U2,U3,X,Z)

        X_intv = np.asarray(X)
        Y_intv = np.asarray(Y)
        Z_intv= np.asarray(Z)

        Intv_X = pd.DataFrame(X_intv)
        Intv_Y = pd.DataFrame(Y_intv)
        Intv_Z = pd.DataFrame(Z_intv)

        Intv_XY = pd.concat([Intv_X, Intv_Y], axis=1)
        Intv_XY.columns = ['X', 'Y']
        Intv_Z.columns = range(self.dim)

        Intv = pd.concat([Intv_XY, Intv_Z], axis=1)
        return Intv

if __name__ == "__main__":
    D = 2
    N = 1000
    seed_num = 1

    datagen = DataGen(D,N,seed_num)
    datagen.gen_U()
    U1 = datagen.U1
    U2 = datagen.U2
    U3 = datagen.U3

    Z = datagen.fZ(U1,U2)

    coef_xz = np.reshape(1 * np.random.rand(D), (D, 1))
    coef_u1x = np.reshape(1 * np.random.rand(D), (D, 1))
    coef_u3x = np.reshape(1 * np.random.rand(D), (D, 1))

    X = datagen.fX(U1,U3,Z)
    Y = datagen.fY(U2,U3,X,Z)
    # X_poly = datagen.poly1(Z)

    OBS = datagen.obs_data()
    PolyIntv = datagen.poly_intv_data()

