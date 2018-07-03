import numpy as np
from DataGen import DataGen
from StoExp import StoExp

class JZ_bound(object):
    def __init__(self):
        return None

    def sepOBS(self, OBS, D):
        X = OBS['X']
        Y = OBS['Y']
        Z = OBS[list(range(D))]
        return [X, Y, Z]

    def JZ_bounds(self, pl, OBS, D, N):
        [X, Y, Z] = self.sepOBS(OBS, D)
        pl_proba = pl.predict_proba(Z)
        pi_val = []

        for idx in range(N):
            xi = int(X[idx])
            proba_elem = pl_proba[idx]
            pi_val.append(proba_elem[xi])
        pi_val = np.array(pi_val)

        Li = np.mean(Y * pi_val)
        Hi = 1 + Li - np.mean(pi_val)
        return [Li, Hi]


if __name__ == "__main__":
    # Parameter configuration
    seed_num = np.random.randint(10000000)
    D = 3
    N = 3000
    T = 5000

    # Generating Observation data
    datagen = DataGen(D,N,seed_num)

    ## Note that Z is binary for simplicity
    OBS = datagen.obs_data()
    [X,Y,Z] = sepOBS(OBS, D)

    ## Example of Policy
    Sto = StoExp(D)
    poly_logit = Sto.Logit(X,Z)
    [Li, Hi] = JZ_bounds(poly_logit, OBS, D, N)

    ## Policy intervention answer
    poly_INTV = datagen.poly_intv_data(poly_logit)
    Y_pi = np.mean(poly_INTV['Y'])



