import pandas as pd
from sklearn import preprocessing
import numpy as np
import copy

# Binarize
def IST_LabelEncoder(IST, ColName):
    le = preprocessing.LabelEncoder()

    COL = copy.copy(IST[ColName])
    list_label = list(pd.unique(COL))
    le.fit(list_label)

    COL = le.transform(COL)

    IST[ColName] = COL
    return IST

def ThreeCategorize(df):
    df_copy = copy.copy(df)
    df_q = list(df.quantile([0.33,0.66,0.99]))

    df_copy[df_copy <= df_q[0]] = 0
    df_copy[(df_copy> df_q[0]) & (df_copy<= df_q[1])] = 1
    df_copy[(df_copy > df_q[1])] = 2

    return df_copy

def LabelRecover(df, df_orig, colname):
    print( pd.concat([df[colname],df_orig[colname]],axis=1) )

def ContNormalization(df,colname):
    df_col = copy.copy(df[colname])
    df_col = (df_col - min(df_col))/(max(df_col) - min(df_col))
    return df_col

def ObsEffect(df):
    return [ np.mean(df[(df['RXASP'] == 0)]['FDEAD']), np.mean(df[(df['RXASP'] == 1)]['FDEAD']) ]


# Data load
np.random.seed(1)
IST = pd.read_csv('IST.csv')

# Randomization data selection
list(IST.columns.values)
chosen_variables = ['SEX','AGE','RSLEEP','RATRIAL','RCONSC','RDELAY',
                    'RVISINF','RHEP24','RASP3','RSBP',
                    'RDEF1','RDEF2','RDEF3','RDEF4','RDEF5','RDEF6','RDEF7','RDEF8',
                    'STYPE','RXASP','RXHEP',
                    'FDEAD', 'EXPD14'
                    ]

# Pre-select
## If RATRIAL is empty, then none
IST = IST[chosen_variables]
IST = IST.loc[pd.isnull(IST['RATRIAL']) == False]
IST = IST.loc[(IST['FDEAD'] != 'U')]
IST = IST.dropna()

IST_orig = copy.copy(IST)

## Binarize
discrete_variables = ['SEX','RSLEEP','RATRIAL','RCONSC',
                    'RVISINF','RHEP24','RASP3',
                    'RDEF1','RDEF2','RDEF3','RDEF4','RDEF5','RDEF6','RDEF7','RDEF8',
                    'STYPE','RXASP','RXHEP',
                    'FDEAD'
                    ]

for disc_val in discrete_variables:
    IST = IST_LabelEncoder(IST,disc_val)

outcome = IST['FDEAD'] - IST['EXPD14']
outcome = pd.DataFrame({'Y':outcome})
IST = pd.concat([IST,outcome],axis=1)

# Discretize the continuous variable
continuous_variable = ['RSBP','AGE','RDELAY'] # list(set(chosen_variables) - set(discrete_variables))
IST['AGE'] = ThreeCategorize(IST['AGE'])
IST['RSBP'] = ThreeCategorize(IST['RSBP'])
IST['RDELAY'] = ThreeCategorize(IST['RDELAY'])

AGE_norm = ContNormalization(IST_orig,'AGE')
BP_norm = ContNormalization(IST_orig,'RSBP')
Delay_norm = ContNormalization(IST_orig,'RDELAY')

# Sampling rule
## Male the higher,
## Older the higher,
## ATRIAL the higher,
## Nonconcious the higher
## Infarct the higher
## High BP the higher,
## Longer delay the higher

### Sampling formula
# Prob = 0.5 + 0.5( 0.1*SEX + 0.1*AGE + 0.2*ATL + 0.2*RSL + 0.2*INF + 0.2*BP )
N = len(IST)
age_coef = 0.05
bp_coef = 0.05
delay_coef = 0.05
sex_coef = 0.05
atl_coef = 0.3
slp_coef = 0.2
inf_coef = 0.3

coefs = [age_coef, bp_coef, delay_coef, sex_coef, atl_coef, slp_coef, inf_coef]

sample_list = []
baseline_prob = 0.0
weighted_prob = 0.6
treatment_prob = 0.4

for idx in range(N):
    elem = IST.iloc[idx]

    elem_age = AGE_norm.iloc[idx]
    elem_BP = BP_norm.iloc[idx]
    elem_DELAY = Delay_norm.iloc[idx]
    elem_sex = elem['SEX']
    elem_ATL = elem['RATRIAL']
    elem_SLP = elem['RSLEEP']
    elem_INF = elem['RVISINF']

    elem_treat = elem['RXASP']

    elems = [elem_age,elem_BP, elem_DELAY, elem_sex, elem_ATL, elem_SLP, elem_INF]

    prob = baseline_prob + weighted_prob * np.dot(coefs, elems) + treatment_prob * (1-elem_treat)
    if np.random.binomial(1, prob) == 0:
        continue
    else:
        sample_list.append(elem)

Sample = pd.DataFrame(sample_list)
# prob_death_p10 = list(Sample['EXPD14'].quantile([0.2]))[0]
# Sample = Sample[Sample['EXPD14'] > prob_death_p10]

# prob_death_p90 = list(Sample['EXPD14'].quantile([0.9]))[0]
# Sample = Sample[Sample['EXPD14'] < prob_death_p90]



print(ObsEffect(IST))
print(ObsEffect(Sample))
print([np.mean(IST['RXASP']),np.mean(Sample['RXASP'])])
print(len(Sample))


