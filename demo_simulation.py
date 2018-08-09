import pandas as pd
from sklearn import preprocessing
import numpy as np
import copy
from scipy import stats

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
    print(pd.unique(df[colname]))
    print(pd.unique(df_orig[colname]))
    # print( pd.concat([df[colname],df_orig[colname]],axis=1) )

def ContNormalization(df,colname):
    df_col = copy.copy(df[colname])
    df_col = (df_col - min(df_col))/(max(df_col) - min(df_col))
    return df_col

def ObsEffect(df,outcome):
    return [np.mean(df[(df['RXASP'] == 0)][outcome]), np.mean(df[(df['RXASP'] == 1)][outcome])]

def diffEffect(df,outcome):
    Yx0, Yx1 = ObsEffect(df,outcome)
    return(np.abs(Yx0-Yx1))

def SigTest(df,X,Y):
    result = stats.ttest_ind(df[df[X] == 0][Y], df[df[X] == 1][Y], equal_var=False)
    return result.pvalue

def ReduceIST(IST):
    IST = IST.loc[pd.isnull(IST['RATRIAL']) == False]
    ## If No dead informatino, then exlucde
    # IST = IST.loc[(IST['FDEAD'] != 'U')]
    ## If No recover informatino, then exlucde
    IST = IST.loc[(IST['FRECOVER']) != 'U']
    ## Patients without taking Heparin, b/c, we are only interested in Aspirin
    # IST = IST.loc[(IST['RXHEP'] == 'N')]
    # ## Patients dead from other causes are exlucded.
    IST = IST.loc[(IST['DEAD7'] == 0) & (IST['DEAD8'] == 0)]
    ## Patients only complied
    # IST = IST.loc[(IST['CMPLASP'] == 'Y')]

    chosen_variables = ['SEX', 'AGE',
                        'RSLEEP', 'RATRIAL', 'RCONSC', 'RDELAY', 'RVISINF', 'RSBP',
                        'RXASP',
                        'FRECOVER',
                        'EXPDD'
                        ]
    IST = IST[chosen_variables]
    IST = IST.dropna()

    IST = IndexifyDisc(IST)

    outcome = 1 * IST['FRECOVER'] - (1 - 1 * IST['EXPDD'])
    outcome = (outcome + 1) / 2
    outcome = pd.DataFrame({'Y': outcome})
    IST = pd.concat([IST, outcome], axis=1)
    return IST

def IndexifyDisc(IST):
    discrete_variables = ['SEX', 'RSLEEP', 'RATRIAL', 'RCONSC',
                          'RVISINF', 'RXASP',
                          'FRECOVER'
                          ]

    for disc_val in discrete_variables:
        IST = IST_LabelEncoder(IST, disc_val)
    return IST

def ContToDisc(IST):
    # Discretize the continuous variable
    continuous_variable = ['RSBP', 'AGE', 'RDELAY']  # list(set(chosen_variables) - set(discrete_variables))
    IST['AGE'] = ThreeCategorize(IST['AGE'])
    IST['RSBP'] = ThreeCategorize(IST['RSBP'])
    IST['RDELAY'] = ThreeCategorize(IST['RDELAY'])
    return IST

def SeedFinding(IST, sample_N, alpha=0.01):
    prev_sig = 10
    iter_idx = 0
    possible_case = 2 ** 32 - 1
    remember_seed = 0

    while 1:
        iter_idx += 1
        seed_num = np.random.randint(possible_case)
        np.random.seed(seed_num)
        Sample = IST.sample(n=sample_N)
        sig_sample = SigTest(Sample,'RXASP','Y')
        if sig_sample < prev_sig:
            prev_sig = sig_sample
            remember_seed = seed_num

        if iter_idx%500 == 0:
            print(seed_num, sig_sample, prev_sig, remember_seed, iter_idx)

        if sig_sample < alpha:
            remember_seed = seed_num
            break

        if iter_idx > possible_case:
            print("all investigated")
            break
    return remember_seed


def GenEXP(IST,sample_N = 10000, remember_seed = 3141693719):
    np.random.seed(remember_seed)
    EXP = IST.sample(n=sample_N)
    return EXP

def GenOBS(EXP, seed_obs = 1):
    np.random.seed(seed_obs)
    weight_sick = 0.05
    weight_treatment = 0.95

    sample_list = []

    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_EXPD = elem['EXPDD']
        elem_treat = elem['RXASP']

        if elem_treat == 0:
            if elem_EXPD < 0.7:
                prob = 0.1
            else:
                prob = 0.9
        else:
            if elem_EXPD < 0.7:
                prob = 0.9
            else:
                prob = 0.1

        selection_prob = np.dot([weight_sick, weight_treatment],[prob, 1-elem_treat])

        if np.random.binomial(1, selection_prob) == 0:
            continue
        else:
            sample_list.append(elem)

    OBS = pd.DataFrame(sample_list)
    return OBS

def HideCovarOBS(EXP,OBS):
    selected_covariates = ['AGE', 'RATRIAL', 'RVISINF', 'RXASP', 'Y']

    ## Resulting dataset
    EXP = EXP[selected_covariates]
    OBS = OBS[selected_covariates]
    return [EXP,OBS]

def ComputeBound(OBS,X):
    Lx0 = np.mean((OBS[X] == 0) * OBS['Y'])
    Lx1 = np.mean((OBS[X] == 1) * OBS['Y'])

    Hx0 = Lx0 + np.mean((OBS[X] == 1))
    Hx1 = Lx1 + np.mean((OBS[X] == 0))

    return [[Lx0,Lx1],[Hx0,Hx1]]

def GroundTruth(EXP,X):
    Ux0 = np.mean(EXP[EXP[X]==0]['Y'])
    Ux1 = np.mean(EXP[EXP[X]==1]['Y'])
    return [Ux0, Ux1]

def CheckCase2(HB,U):
    hx0, hx1 = HB
    Ux0, Ux1 = U

    if Ux0 < Ux1:
        if Ux1 < hx0:
            return True
        else:
            return False
    else:
        if Ux0 < hx1:
            return True
        else:
            return False

# Data load
IST = pd.read_csv('IST.csv')
X = 'RXASP'

IST = ReduceIST(IST)

IST = IndexifyDisc(IST)
IST = ContToDisc(IST)

EXP = GenEXP(IST)
OBS = GenOBS(EXP)

print(ObsEffect(EXP,'Y'))
print(ObsEffect(OBS,'Y'))

EXP,OBS = HideCovarOBS(EXP,OBS)
LB,HB = ComputeBound(OBS,X)
U = GroundTruth(EXP,X)
TF_Case2 = CheckCase2(HB,U)

lx0, lx1 = LB
hx0, hx1 = HB
Ux0, Ux1 = U

print([lx0,Ux0,hx0])
print([lx1,Ux1,hx1])



# # Select necessary columns
# ## If No Atrial information, then exclude those
# IST = IST.loc[pd.isnull(IST['RATRIAL']) == False]
# ## If No dead informatino, then exlucde
# # IST = IST.loc[(IST['FDEAD'] != 'U')]
# ## If No recover informatino, then exlucde
# IST = IST.loc[(IST['FRECOVER'])!='U']
# ## Patients without taking Heparin, b/c, we are only interested in Aspirin
# # IST = IST.loc[(IST['RXHEP'] == 'N')]
# # ## Patients dead from other causes are exlucded.
# IST = IST.loc[(IST['DEAD7'] == 0) & (IST['DEAD8']==0)]
# ## Patients only complied
# # IST = IST.loc[(IST['CMPLASP'] == 'Y')]
#
# chosen_variables = ['SEX','AGE',
#                     'RSLEEP','RATRIAL','RCONSC','RDELAY','RVISINF','RSBP',
#                     'RXASP',
#                     'FRECOVER',
#                     'EXPDD'
#                     ]
# IST = IST[chosen_variables]
# IST = IST.dropna()
# IST_orig = copy.copy(IST)

## Binarize
# discrete_variables = ['SEX','RSLEEP','RATRIAL','RCONSC',
#                     'RVISINF','RXASP',
#                     'FRECOVER'
#                     ]
#
# for disc_val in discrete_variables:
#     IST = IST_LabelEncoder(IST,disc_val)

# outcome = 1*IST['FRECOVER'] - (1-1*IST['EXP14'])
# outcome = 1*IST['FRECOVER'] - (1-1*IST['EXPDD'])
# outcome = (outcome + 1)/2
# outcome = pd.DataFrame({'Y':outcome})
# IST = pd.concat([IST,outcome],axis=1)

# Random sampling for EXP generation
# N = len(IST)
# sample_N = 10000
# prev_sig = 10
# iter_idx = 0
# possible_case = 2**32-1
# remember_seed = 0

# while 1:
#     iter_idx += 1
#     seed_num = np.random.randint(possible_case)
#     np.random.seed(seed_num)
#     Sample = IST.sample(n=sample_N)
#     sig_sample = SigTest(Sample,'RXASP','Y')
#     if sig_sample < prev_sig:
#         prev_sig = sig_sample
#         remember_seed = seed_num
#
#     if iter_idx%500 == 0:
#         print(seed_num, sig_sample, prev_sig, remember_seed, iter_idx)
#
#     if sig_sample < 0.01:
#         remember_seed = seed_num
#         break
#
#     if iter_idx > possible_case:
#         print("all investigated")
#         break

# remember_seed = 3141693719
#
# # Discretize the continuous variable
# continuous_variable = ['RSBP','AGE','RDELAY'] # list(set(chosen_variables) - set(discrete_variables))
# IST['AGE'] = ThreeCategorize(IST['AGE'])
# IST['RSBP'] = ThreeCategorize(IST['RSBP'])
# IST['RDELAY'] = ThreeCategorize(IST['RDELAY'])

# AGE_norm = ContNormalization(IST_orig,'AGE')
# BP_norm = ContNormalization(IST_orig,'RSBP')
# Delay_norm = ContNormalization(IST_orig,'RDELAY')

# np.random.seed(remember_seed)
# # remember_seed = 931166988
# # remember_seed = 3141693719
# EXP = IST.sample(n=sample_N)

# iter_idx = 0
# obs_N = round(sample_N / 10)
# remember_seed = 0
# prev_diff = 0
# while 1:
#     iter_idx += 1
#     seed_num = np.random.randint(possible_case)
#     np.random.seed(seed_num)
#     Sample = EXP.sample(n=obs_N)
#     Yx0 = Sample[Sample['RXASP']==0]['Y']
#     Yx1 = Sample[Sample['RXASP']==1]['Y']
#     Ex0 = np.mean(Yx0)
#     Ex1 = np.mean(Yx1)
#
#     if (len(Yx0) < obs_N/10 or len(Yx1) < obs_N/10) and Ex0 > Ex1:
#         remember_seed = seed_num
#         break
#
#     if iter_idx%500 == 0:
#         print(seed_num, remember_seed,Ex0 - Ex1, iter_idx)
#
#     if iter_idx > possible_case:
#         print("all investigated")
#         break




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

# age_coef = 0
# bp_coef = 0
# delay_coef = 0
# sex_coef = 0
# atl_coef = 0.4
# slp_coef = 0.1
# inf_coef = 0.5
#
# coefs = [age_coef, bp_coef, delay_coef, sex_coef, atl_coef, slp_coef, inf_coef]
#
# sample_list = []
# baseline_prob = 0.4
# weighted_prob = 0.4
# EXPD_prob = 0.2
#
# np.random.seed(1)
# for idx in range(len(EXP)):
#     elem = EXP.iloc[idx]
#
#     elem_age = elem['AGE']/2
#     elem_BP = elem['RSBP']/2
#     elem_DELAY = elem['RDELAY']/2
#     elem_sex = elem['SEX']
#     elem_ATL = elem['RATRIAL']
#     elem_SLP = elem['RSLEEP']
#     elem_INF = elem['RVISINF']
#
#     elem_EXPD = elem['EXPDD']
#     elem_treat = elem['RXASP']
#
#     elems = [elem_age,elem_BP, elem_DELAY, elem_sex, elem_ATL, elem_SLP, elem_INF]
#
#     if elem_treat == 0:
#         if elem_EXPD < 0.7:
#             prob = 0.1
#         else:
#             prob = 0.9
#     else:
#         if elem_EXPD < 0.7:
#             prob = 0.9
#         else:
#             prob = 0.1
#
#     # prob = baseline_prob + \
#     #        weighted_prob * (1-np.dot(coefs, elems)) + \
#     #        EXPD_prob * (1-elem_EXPD)
#
#     if np.random.binomial(1, 0.05 * prob + 0.95*(1-elem_treat)) == 0:
#         continue
#     else:
#         sample_list.append(elem)
#
# OBS = pd.DataFrame(sample_list)

# prob_death_p10 = list(Sample['EXPD14'].quantile([0.2]))[0]
# Sample = Sample[Sample['EXPD14'] > prob_death_p10]

# prob_death_p90 = list(Sample['EXPD14'].quantile([0.9]))[0]
# Sample = Sample[Sample['EXPD14'] < prob_death_p90]


# print(ObsEffect(IST,'Y'))
# print(ObsEffect(Sample,'Y'))
# print()
# print(ObsEffect(IST,'FRECOVER'))
# print(ObsEffect(Sample,'FRECOVER'))
# print()
#
# print([np.mean(IST['RXASP']),np.mean(Sample['RXASP'])])
# print(len(Sample))

# If sampling rule is good enough, then hide some Z.
# selected_covariates = ['AGE','RATRIAL','RVISINF','RXASP','Y']
#
# ## Resulting dataset
# EXP = EXP[selected_covariates]
# OBS = OBS[selected_covariates]

# Check Case 2
# # Lx0 = np.mean((OBS['RXASP']==0) * OBS['Y'])
# # Lx1 = np.mean((OBS['RXASP']==1) * OBS['Y'])
# #
# # Hx0 = Lx0 + np.mean((OBS['RXASP']==1))
# # Hx1 = Lx1 + np.mean((OBS['RXASP']==0))
#
# # Lx0 = np.mean(OBS[OBS['RXASP']==0]['Y']) * len(OBS[OBS['RXASP']==0])/len(OBS)
# # Lx1 = np.mean(OBS[OBS['RXASP']==1]['Y']) * len(OBS[OBS['RXASP']==1])/len(OBS)
# #
# # Hx0 = Lx0 + len(OBS[OBS['RXASP']==1])/len(OBS)
# # Hx1 = Lx1 + len(OBS[OBS['RXASP']==0])/len(OBS)
# #
# Ux0 = np.mean(EXP[EXP['RXASP']==0]['Y'])
# Ux1 = np.mean(EXP[EXP['RXASP']==1]['Y'])
#
# print([Lx0,Ux0,Hx0])
# print([Lx1,Ux1,Hx1])