import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class DPFit(object):
    def __init__(self, Y, init_compo):
        '''
        :param Y: Outcome variable, Obs['Y'] Or Intv['Y']
        '''
        self.Y = Y
        self.init_compo = init_compo

    def preprocess(self, Y):
        if Y.shape[1] >= Y.shape[0] or pd.isnull(Y.shape[1]):
            Y = np.matrix(Y).T
        else:
            pass
        return Y

    def Fit(self, Y):
        Y = np.reshape(Y, (len(Y), 1))
        Y = self.preprocess(Y)
        dpgmm= mixture.BayesianGaussianMixture(
            n_components=self.init_compo, weight_concentration_prior= 1,
            max_iter=100, tol=1e-8 ).fit(Y)
        return dpgmm

    def DP_samples(self, dpgmm):
        Y_sampled = dpgmm.sample(len(self.Y))[0]
        return Y_sampled

    def DPPlot(self, Y, Y_sampled):
        f = plt.figure()

        X_int = np.reshape(Y, (len(Y), 1))
        X_int = self.preprocess(X_int)
        int_plot = f.add_subplot(211)
        X_int_kde = np.ndarray.flatten(X_int)
        densityfun = gaussian_kde(X_int_kde)
        x_domain_int = np.linspace(min(X_int_kde) - abs(max(X_int_kde)), max(X_int_kde) + abs(max(X_int_kde)),
                                   len(Y))
        int_plot.plot(x_domain_int, densityfun(x_domain_int))
        int_plot.hist(X_int, 100, normed=True)
        int_plot.set_title('Original')

        sim_plot = f.add_subplot(212)
        X_sim = np.ndarray.flatten(Y_sampled)
        # X_sim= [item for sublist in X_sim for item in sublist]
        sim_density = gaussian_kde(X_sim)
        # sim_x_domain = np.linspace(min(X_sim) - 5, max(X_sim) + 5, self.num_data)
        sim_plot.plot(x_domain_int, sim_density(x_domain_int))
        sim_plot.hist(X_sim, 100, normed=True)
        sim_plot.set_title('Sampled by DP simluation')
        return f

    def Conduct(self):
        self.dpgmm = self.Fit(self.Y)
        # Y_sampled = self.DP_samples(self.dpgmm)
        # self.DPPlot(self.Y, Y_sampled)
