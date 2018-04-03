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
    # np.random.seed(180402)
    np.random.seed(1234)
    def __init__(self,D,N,Ns,Mode):
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
        self.num_intv = Ns
        self.Mode = Mode # easy / crazy

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

        if self.Mode == 'crazy':
            U1 = self.weird_projection(U1)
            U2 = self.weird_projection(U2)
            U3 = self.weird_projection(U3)
            #
            # U1 = (U1 - np.mean(U1, axis=0)) / np.var(U1)
            # U2 = (U2 - np.mean(U2, axis=0)) / np.var(U2)
            # U3 = (U3 - np.mean(U3, axis=0)) / np.var(U3)

        self.U1 = U1
        self.U2 = U2
        self.U3 = U3

    def gen_Z(self):
        if self.Mode == 'easy':
            Z = -self.U1 + 2*self.U2
        elif self.Mode == 'crazy':
            # Z = np.exp(np.abs(self.U1)+1) + np.power(self.U2, 3)
            Z = np.exp(self.U1) + np.exp(self.U2)
            # self.Z = ((Z - np.mean(Z, axis=0)) / np.var(Z))
        # self.Z = ((Z - np.mean(Z, axis=0)) / np.var(Z))
        self.Z = Z

    def gen_X(self):
        coef_xz = np.reshape(-2 * np.random.rand(self.dim), (self.dim, 1))
        coef_u1x = np.reshape(-2 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3x = np.reshape(5 * np.random.rand(self.dim), (self.dim, 1))

        U1 = np.matrix(self.U1)
        U3 = np.matrix(self.U3)
        Z = np.matrix(self.Z)

        if self.Mode == 'easy':
            X_obs = sp.expit( Z*coef_xz + U3 * coef_u3x + U1 * coef_u1x  )
            X_obs = (X_obs - min(X_obs))/(max(X_obs) - min(X_obs))
            X_obs = np.round(X_obs,0)
            self.X_obs = self.intfy(np.array(np.matrix.flatten(X_obs).tolist()[0]))

        elif self.Mode == 'crazy':
            X_obs = sp.expit( np.power( np.abs(U1 * coef_u1x), 1)  - np.power(U3 * coef_u3x , 1) + (np.exp(Z * coef_xz)) )

            X_obs = (X_obs - min(X_obs)) / (max(X_obs) - min(X_obs))
            X_obs = np.round(X_obs, 0)
            # self.X_obs = X_obs
            self.X_obs = self.intfy(np.array(np.matrix.flatten(X_obs).tolist()[0]))
        self.X_intv = self.intfy(np.asarray([0] * int(self.num_obs / 2) +
                                            [1] * int(self.num_obs / 2)))

    def gen_Y(self):
        coef_zy = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u2y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))
        coef_u3y = np.reshape(1 * np.random.rand(self.dim), (self.dim, 1))

        U2 = np.matrix(self.U2)
        U3 = np.matrix(self.U3)
        Z = np.matrix(self.Z)
        X_obs = np.matrix(self.X_obs)
        X_intv = np.matrix(self.X_intv)


        if self.Mode == 'easy':
            # Parametrization determination
                ## Case 3: constant to 1
                ## Case 2: constant to 1.5
                ## Case 1: constant to 2
            # Y = U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy
            # Y = 100*(Y-np.mean(Y))/np.var(Y)
            # Y += (2*np.array(X_obs.T)-1)
            #
            # Y_intv = U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy
            # Y_intv = 100*(Y_intv - np.mean(Y)) / np.var(Y)
            # Y_intv += (2 * np.array(X_intv.T) - 1)

            Y =  100*U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy + 700 * (2*np.array(X_obs.T)-1)
            Y_intv = 100* U2 * coef_u2y + U3 * coef_u3y + Z * coef_zy + 700 * (2*np.array(X_intv.T)-1)
            print(Y)

        elif self.Mode == 'crazy':
            Y = 5*np.array(np.sin(U2 * coef_u2y)) + \
                np.array(-1 * np.array(1 * np.power(U3 * coef_u3y, 2)) *
                         (2*np.array(X_obs.T)-1)) + \
                np.array(np.power(np.abs(Z * coef_zy), 0.2)) * np.array(2 * np.array(X_obs.T) - 1)

            print(Y)

            #
            # # Y = 100 * ((Y - np.mean(Y, axis=0)) / np.var(Y))
            #
            Y_intv = 5*np.array(np.sin(U2 * coef_u2y)) + \
                     np.array(-1 * np.array(1* np.power(U3 * coef_u3y,2)) *
                              (2*np.array(X_intv.T)-1)) + \
                     np.array(np.power(np.abs(Z * coef_zy),0.2) )  * np.array(2 * np.array(X_intv.T) - 1)

            # Y_intv = 100 * ((Y_intv - np.mean(Y, axis=0)) / np.var(Y))

            # Y = np.array(np.sin(U2 * coef_u2y)) * \
            #     np.array(-2 * np.array(np.power(U3 * coef_u3y,2)) +
            #          +5*np.exp( 3*(2*np.array(X_obs.T) -1)) * (2*np.array(X_obs.T) -1 )) - \
            #         20 * np.array(2 * np.array(X_obs.T) - 1) + \
            #     np.array(np.power(np.abs(Z * coef_zy + 1),1))
            #
            # print(Y)
            # Y_intv = np.array(np.sin(U2 * coef_u2y)) * \
            #          np.array(-2 * np.array(np.power(U3 * coef_u3y,2)) +
            #               +5*np.exp( 3*(2*np.array(X_intv.T)-1)) * (2*np.array(X_intv.T) -1 )) - \
            #          20*np.array( 2*np.array(X_intv.T)-1 ) + \
            #          np.array(np.power(np.abs(Z * coef_zy + 1),1))
        # self.Y = Y
        # self.Y_intv = Y_intv
        self.Y = Y
        self.Y_intv = Y_intv

        # self.Y = ((Y - min(Y)) / (max(Y)-min(Y)))
        # self.Y_intv = ((Y_intv - min(Y_intv)) / (max(Y_intv) - min(Y_intv)))


    def structure_data(self):
        X = self.X_obs
        X_intv = self.X_intv
        Y = self.Y
        Y = (Y-min(Y))/(max(Y)-min(Y))
        Y_intv = self.Y_intv
        Y_intv = (Y_intv - min(Y_intv)) / (max(Y_intv) - min(Y_intv))
        Z = self.Z


        X_obs = np.asarray(X)
        Y_obs = np.asarray(Y)
        Z_obs = np.asarray(Z)

        Obs_X = pd.DataFrame(X_obs)
        Obs_Y = pd.DataFrame(Y_obs)
        Obs_Z = pd.DataFrame(Z_obs)

        Obs_XY = pd.concat([Obs_X, Obs_Y], axis=1)
        Obs_XY.columns = ['X', 'Y']
        Obs_Z.columns = range(self.dim)

        X_intv = np.asarray(X_intv)
        Y_intv = np.asarray(Y_intv)
        Intv_X = pd.DataFrame(X_intv)
        Intv_Y = pd.DataFrame(Y_intv)
        Intv_Z = pd.DataFrame(Z_obs)

        Intv_XY = pd.concat([Intv_X, Intv_Y], axis=1)
        Intv_XY.columns = ['X', 'Y']
        Intv_Z.columns = range(self.dim)

        sample_indces_x1 = np.random.choice(list(range(0, int(self.num_obs / 2))), int(self.num_intv / 2), replace=False)
        sample_indces_x0 = np.random.choice(list(range(int(self.num_obs / 2), self.num_obs)), int(self.num_intv / 2), replace=False)
        sample_indices = np.asarray(list(sample_indces_x1) + list(sample_indces_x0))

        X_sintv = X_intv[sample_indices]
        Y_sintv = Y_intv[sample_indices]
        Z_sintv = Z_obs[sample_indices]
        SIntv_X = pd.DataFrame(X_sintv)
        SIntv_Y = pd.DataFrame(Y_sintv)
        SIntv_Z = pd.DataFrame(Z_sintv)

        SIntv_XY = pd.concat([SIntv_X, SIntv_Y], axis=1)
        SIntv_XY.columns = ['X', 'Y']
        SIntv_Z.columns = range(self.dim)

        self.Obs = pd.concat([Obs_XY, Obs_Z], axis=1)
        self.Intv = pd.concat([Intv_XY, Intv_Z], axis=1)
        self.Intv_S = pd.concat([SIntv_XY, SIntv_Z], axis=1)

    def data_gen(self):
        self.gen_U()
        self.gen_Z()
        self.gen_X()
        self.gen_Y()
        self.structure_data()

if __name__ == "__main__":
    D = 3
    N = 1000
    Ns = 20
    Mode = 'crazy'

    datagen = DataGen(D,N,Ns,Mode)
    datagen.data_gen()

    Obs = datagen.Obs
    Intv = datagen.Intv

    plt.figure('Obs')
    plt.hist(Obs['Y'],50)

    plt.figure('intv')
    plt.hist(Intv['Y'], 50)