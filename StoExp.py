import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

class StoExp(object):
    def __init__(self, D):
        self.D = D

    def Construct_XZ(self, DF):
        Z = DF[list(range(self.D))]
        X = DF['X']
        return [Z, X]

    def Trun_by_Y(self, DF, goodprop, criterion):
        DF_good = list(DF[DF['Y'] > criterion].index)
        DF_bad = list(DF[DF['Y'] <= criterion].index)

        N = min(len(DF_good), len(DF_bad))
        N_good = round(N*goodprop)
        N_bad = N-N_good
        sampled_good_idx = random.sample(DF_good, N_good)
        sampled_bad_idx = random.sample(DF_bad, N_bad)
        sampled_idx = sampled_good_idx + sampled_bad_idx

        return DF.ix[sampled_idx]

    def Logit(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        logit_reg = LogisticRegression()
        logit_reg.fit(Z_train, X_train)
        return logit_reg

    def SVM_linear(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        svmcls = svm.SVC(kernel='linear', probability=True)
        svmcls.fit(Z_train, X_train)
        return svmcls

    def SVM_rbf(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        svmcls = svm.SVC(kernel='rbf', probability=True)
        svmcls.fit(Z_train, X_train)
        return svmcls

    def MLP(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        mlpcls = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
        mlpcls.fit(Z_train, X_train)
        return mlpcls

    def LDA(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        ldacls = LinearDiscriminantAnalysis()
        ldacls.fit(Z_train, X_train)
        return ldacls


    def XGB(self, X, Z):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        xgbcls = XGBClassifier(max_depth=10)
        xgbcls.fit(Z_train, X_train)
        return xgbcls

    def Compute_C(self, obspolicy, target_policy, Z):
        obsprob = obspolicy.predict_proba(Z)
        targetprob = target_policy.predict_proba(Z)
        N = len(Z)
        sum_entropy = 0

        for idx in range(N):
            sum_entropy += obsprob[idx][0] * np.log(1/(targetprob[idx][0])) + obsprob[idx][1] * np.log(1/(targetprob[idx][1]))
        return sum_entropy / N
