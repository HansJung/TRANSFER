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

def BinaryCategorize(df):
    df_copy = copy.copy(df)
    df_q = list(df.quantile([0.5]))

    df_copy[df_copy <= df_q[0]] = 0
    df_copy[df_copy > df_q[0]] = 1

    return df_copy

def LabelRecover(df, df_orig, colname):
    print(pd.unique(df[colname]))
    print(pd.unique(df_orig[colname]))
    # print( pd.concat([df[colname],df_orig[colname]],axis=1) )

def ContNormalization(df,colname):
    df_col = copy.copy(df[colname])
    df_col = (df_col - min(df_col))/(max(df_col) - min(df_col))
    return df_col

def ComputeEffect(df,outcome):
    return [np.mean(df[(df['RXASP'] == 0)][outcome]), np.mean(df[(df['RXASP'] == 1)][outcome])]

def diffEffect(df,outcome):
    Yx0, Yx1 = ComputeEffect(df,outcome)
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
    IST['AGE'] = BinaryCategorize(IST['AGE'])
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
    weight_sick = 0.01
    weight_treatment = 0.99

    sample_list = []

    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_EXPD = elem['EXPDD']
        elem_treat = elem['RXASP']

        if elem_treat == 0:
            if elem_EXPD < 0.7:
                prob = 0.2
            else:
                prob = 0.8
        else:
            if elem_EXPD < 0.7:
                prob = 0.8
            else:
                prob = 0.2

        selection_prob = np.dot([weight_sick, weight_treatment],[prob, 1-elem_treat])

        if np.random.binomial(1, selection_prob) == 0:
            continue
        else:
            sample_list.append(elem)

    OBS = pd.DataFrame(sample_list)
    return OBS

def HideCovarDF(DF):
    selected_covariates = ['AGE', 'SEX', 'RXASP', 'Y']

    ## Resulting dataset
    DF = DF[selected_covariates]
    return DF

def HideCovarOBS(EXP,OBS):
    selected_covariates = ['AGE', 'SEX', 'RXASP', 'Y']

    ## Resulting dataset
    EXP = EXP[selected_covariates]
    OBS = OBS[selected_covariates]
    return [EXP,OBS]

def ComputeBound(OBS,X):
    Px0 = len(OBS[OBS[X]==0])/len(OBS)
    Px1 = len(OBS[OBS[X]==1])/len(OBS)

    # Lx0 = np.mean((OBS[X] == 0) * OBS['Y'])
    # Lx1 = np.mean((OBS[X] == 1) * OBS['Y'])

    EY_x0 = np.mean(OBS[OBS[X] == 0]['Y'])
    EY_x1 = np.mean(OBS[OBS[X] == 1]['Y'])
    Lx0 = EY_x0 * Px0
    Lx1 = EY_x1 * Px1

    Hx0 = Lx0 + Px1
    Hx1 = Lx1 + Px0

    return [[Lx0,Lx1],[Hx0,Hx1]]

def EmpiricalComputeBound(OBS,X,delta):
    N = len(OBS)
    delta = delta/2
    fn = np.sqrt(((2 * N) ** (-1)) * (np.log(4) - np.log(delta)))

    Px0 = len(OBS[OBS[X] == 0]) / len(OBS)
    Px1 = len(OBS[OBS[X] == 1]) / len(OBS)

    Lx0 = np.mean((OBS[X] == 0) * OBS['Y'])
    Lx1 = np.mean((OBS[X] == 1) * OBS['Y'])

    Lx0 = max(0,Lx0 - fn)
    Lx1 = max(0,Lx1 - fn)

    Hx0 = Lx0 + Px1
    Hx1 = Lx1 + Px0

    Hx0 = min(Hx0 + fn,1)
    Hx1 = min(Hx1 + fn,1)

    return [[Lx0, Lx1], [Hx0, Hx1]]

def GroundTruth(EXP,X):
    Ux0 = np.mean(EXP[EXP[X]==0]['Y'])
    Ux1 = np.mean(EXP[EXP[X]==1]['Y'])
    return [Ux0, Ux1]

def CheckCase2(HB,U):
    hx0, hx1 = HB
    Ux0, Ux1 = U

    if Ux0 < Ux1:
        if hx0 < Ux1:
            return True
        else:
            return False
    else:
        if hx1 < Ux0:
            return True
        else:
            return False

def ChangeRXASPtoX(df,idx_X=3):
    df_colnames = list(df.columns)
    df_colnames[idx_X] = 'X'
    df.columns = df_colnames
    return df

def RunGenData(sample_N=12000, remember_seed = 1444260861):
    # Data load
    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST)

    IST = IndexifyDisc(IST)
    IST = ContToDisc(IST)

    # remember_seed = SeedFinding(IST,sample_N=12000, alpha=0.01)

    EXP = GenEXP(IST,sample_N,remember_seed)
    OBS = GenOBS(EXP)

    EXP, OBS = HideCovarOBS(EXP, OBS)
    IST = HideCovarDF(IST)

    return [IST, EXP,OBS]

def QualityCheck(EXP,OBS,X,TF_emp = False,delta=0.01):
    if TF_emp:
        LB, HB = EmpiricalComputeBound(OBS, X,delta)
    else:
        LB, HB = ComputeBound(OBS, X)
    U = GroundTruth(EXP, X)
    TF_Case2 = CheckCase2(HB, U)

    lx0, lx1 = LB
    hx0, hx1 = HB
    Ux0, Ux1 = U

    print([lx0, Ux0, hx0])
    print([lx1, Ux1, hx1])
    return(TF_Case2)

if __name__ == "__main__":
    EXP,OBS = RunGenData()




