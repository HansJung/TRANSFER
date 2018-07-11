import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

class StoExp(object):
    '''
    Set of policies
    '''
    def __init__(self, D):
        self.D = D

    def sepOBS(self, OBS):
        '''
        Separate data frame (X,Y,Z) to X,Y,Z
        :param OBS: (X,Y,Z)
        :param D: dimension of Z
        :return: X,Y,Z
        '''
        X = OBS['X']
        Y = OBS['Y']
        Z = OBS[list(range(self.D))] # multi-dimensional z is indexed by 0,1,2,...,d
        return [X, Y, Z]

    def Construct_XZ(self, OBS):
        '''

        :param DF: Data Frame of (X,Y,Z)
        :return: (X,Z)
        '''
        Z = OBS[list(range(self.D))]
        X = OBS['X']
        return [Z, X]

    def Logit(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a logistic regression model
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        logit_reg = LogisticRegression()
        logit_reg.fit(Z_train, X_train)
        return logit_reg

    def SVM_linear(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a linear SVM
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        svmcls = svm.SVC(kernel='linear', probability=True)
        svmcls.fit(Z_train, X_train)
        return svmcls

    def SVM_rbf(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a RBF SVM
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        svmcls = svm.SVC(kernel='rbf', probability=True)
        svmcls.fit(Z_train, X_train)
        return svmcls

    def MLP(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a Multi-layered perceptron (MLP) neural network
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        mlpcls = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
        mlpcls.fit(Z_train, X_train)
        return mlpcls

    def LDA(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a Linear discriminant analysis
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        ldacls = LinearDiscriminantAnalysis()
        ldacls.fit(Z_train, X_train)
        return ldacls


    def XGB(self, X, Z):
        '''
        :return: pi(X|Z), where pi is a XGboost
        '''
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X, test_size=0.1, random_state=0)
        xgbcls = XGBClassifier(max_depth=10)
        xgbcls.fit(Z_train, X_train)
        return xgbcls

