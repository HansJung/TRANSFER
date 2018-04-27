import numpy as np
import scipy as sp
import pandas as pd
import pickle

data = pickle.load(open('Temp/8971487_5_4.pkl','rb'))

for mu_obs, mu_do in zip(data[1]['aveage_mu'],data[2]['aveage_mu']):
    print(mu_obs,mu_do)