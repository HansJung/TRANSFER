import numpy as np
from DataGen import DataGen
from StoExp import StoExp

class JZ_bound(object):
    '''
    Upper and lower bound (Justin's bound)
    '''
    def __init__(self):
        return None

    def sepOBS(self, OBS, D):
        '''
        Separate data frame (X,Y,Z) to X,Y,Z
        :param OBS: (X,Y,Z)
        :param D: dimension of Z
        :return: X,Y,Z
        '''
        X = OBS['X']
        Y = OBS['Y']
        Z = OBS[list(range(D))] # multi-dimensional z is indexed by 0,1,2,...,d
        return [X, Y, Z]

    def JZ_bounds(self, pl, OBS, D, N):
        '''
        Computing JZ bound for given policy (pl)
        :param pl: given policy
        :param OBS: (X,Y,Z)
        :param D: dimension of Z
        :param N: number of samples
        :return:
        '''
        [X, Y, Z] = self.sepOBS(OBS, D)
        pl_proba = pl.predict_proba(Z) # list of [pi(X=0|zi),pi(X=1|zi)] for each i = 1,2,...,N
        pi_val = [] # list of pi(xi|zi)

        for idx in range(N):
            xi = int(X[idx]) # each xi
            proba_elem = pl_proba[idx] # [pi(X=0|zi),pi(X=1|zi)]
            pi_val.append(proba_elem[xi]) # if xi=0, pi(X=0|zi) // if xi=1, pi(X=1|zi) // i.e., pi(xi|zi)
        pi_val = np.array(pi_val) # list of pi(xi|zi)

        Li = np.mean(Y * pi_val) # approximated lower bound Li
        Hi = 1 + Li - np.mean(pi_val) # approximated upper bound Hi
        return [Li, Hi]


if __name__ == "__main__":
    print('Hello World')



