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

def ComputeEffect(df,X,outcome):
    return [np.mean(df[(df['RXASP'] == 0)][outcome]), np.mean(df[(df['RXASP'] == 1)][outcome])]

def diffEffect(df,outcome):
    Yx0, Yx1 = ComputeEffect(df,outcome)
    return(np.abs(Yx0-Yx1))


def ReduceIST(IST, chosen_variables):
    # Exclude patients not having AF
    # IST = IST.loc[pd.isnull(IST['RATRIAL']) == False]
    # Exclude patients having no recovery information
    IST = IST.loc[(IST['FRECOVER']) != 'U']
    IST = IST.loc[(IST['FDEAD'])!= 'U']
    IST = IST[chosen_variables]
    return IST

def IndexifyDisc(IST, discrete_variables):
    for disc_val in discrete_variables:
        IST = IST_LabelEncoder(IST, disc_val)
    return IST

def ContToDisc(IST,continuous_variables):
    # Discretize the continuous variable
    # continuous_variable = ['RSBP', 'AGE', 'RDELAY']  # list(set(chosen_variables) - set(discrete_variables))
    for cont_val in continuous_variables:
        IST[cont_val] = BinaryCategorize(IST[cont_val])
    return IST

def SigTest(df,X,Y):
    result = stats.ttest_ind(df[df[X] == 0][Y], df[df[X] == 1][Y], equal_var=False)
    return result.pvalue

def SeedFinding(IST, sample_N, alpha=0.001):
    # Find the random seed number
    prev_sig = 2 ** 32 - 1
    iter_idx = 0
    possible_case = 2 ** 32 - 1
    remember_seed = 0

    X = 'RXHEP'

    while 1:
        iter_idx += 1
        # Randomly generating the seed
        seed_num = np.random.randint(possible_case)
        np.random.seed(seed_num)
        # Randomly sample from randomly chosen seed
        Sample = IST.sample(n=sample_N)
        # Test the significance of samples
        sig_sample = SigTest(Sample,X,'Y')
        # Store the minimum significance and the seed number
        if sig_sample < prev_sig:
            prev_sig = sig_sample
            remember_seed = seed_num

        if iter_idx%500 == 0:
            print(seed_num, sig_sample, prev_sig, remember_seed, iter_idx)

        # If the significant level is small enough, then stop
        if sig_sample < alpha:
            remember_seed = seed_num
            break

        if iter_idx > possible_case:
            print("all investigated")
            break
    return remember_seed


def GenEXP(IST,sample_N, remember_seed = 3141693719):
    np.random.seed(remember_seed)
    EXP = IST.sample(n=sample_N)
    return EXP

def GenOBS(EXP,params):
    np.random.seed(1)
    def healthy(x,age,sex,consc,params):
        return np.dot([x,age,sex,consc],params)

    sample_list = []
    for idx in range(len(EXP)-1):
        elem = EXP.iloc[idx]
        elem_EXPD = elem['EXPDD']
        elem_age = elem['AGE']
        elem_sex = elem['SEX']
        elem_consc = elem['RCONSC']
        elem_treat = elem['RXASP']

        # MAKE THIS CODE MORE INTERPRETABLE
        if elem_treat == 0:
            if healthy(elem_treat,elem_age,elem_sex,elem_consc,params) < 0.4:
                prob = 0.01
            else:
                prob = 0.6
        else:
            if healthy(elem_treat, elem_age, elem_sex, elem_consc, params) < 0.35:
                prob = 0.4
            else:
                prob = 0.01

        # Computing the selection probability of patients idx
        # selection_prob = np.dot([weight_sick, weight_treatment],[prob, 1-elem_treat])
        # print(idx, prob, selection_prob)

        if np.random.binomial(1, prob) == 0:
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

def HideCovarOBS(EXP,OBS,selected_covariates):
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

def EmpiricalComputeBound(OBS,X,delta,N):

    # sampleOBS = OBS.sample(n=N)
    sampleOBS = OBS.iloc[:N]
    delta = delta/2
    fn = np.sqrt(((2 * N) ** (-1)) * (np.log(4) - np.log(delta)))

    Px0 = len(sampleOBS[sampleOBS[X] == 0]) / N
    Px1 = len(sampleOBS[sampleOBS[X] == 1]) / N

    Lx0_before = np.mean((sampleOBS[X] == 0) * sampleOBS['Y'])
    Lx1_before = np.mean((sampleOBS[X] == 1) * sampleOBS['Y'])

    Lx0 = max(0,Lx0_before - fn)
    Lx1 = max(0,Lx1_before - fn)

    Hx0 = Lx0_before + Px1
    Hx1 = Lx1_before + Px0

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



def GenAddY(df,necessary_set,params):
    df['Y'] = df[necessary_set[0]] * params[0]
    for idx in range(1,len(necessary_set)):
        df['Y'] = df['Y'] + df[necessary_set[idx]]*params[idx]
    df['Y'] = df['Y'] + np.random.normal(loc=0,scale=0.1,size=len(df))
    maxY = max(df['Y'])
    minY = min(df['Y'])
    df['Y'] = (df['Y']-minY)/(maxY-minY)
    return df


def RunGenData():
    chosen_variables = ['SEX', 'AGE', 'RCONSC',
                        'RXASP', 'EXPDD'
                        ]
    discrete_variables = ['SEX', 'RCONSC', 'RXASP']
    continuous_variables = ['AGE', 'EXPDD']
    necessary_set = ['RXASP', 'AGE', 'SEX', 'RCONSC']
    params = [0.2, -0.2, 0.2, 0.2]

    # Data load
    ## Preprocessing
    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST, chosen_variables)
    IST_orig = copy.copy(IST)
    IST = IndexifyDisc(IST, discrete_variables)
    IST['SEX'] = 1 - IST['SEX']
    IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)

    IST = ContToDisc(IST, continuous_variables)
    EXP = GenAddY(IST, necessary_set, params)
    OBS = GenOBS(EXP, params)
    print("OBS", ComputeEffect(OBS, 'RXASP', 'Y'))
    print("EXP", ComputeEffect(EXP, 'RXASP', 'Y'))
    print('Non-Emp',QualityCheck(EXP,OBS,'RXASP'))
    print('Emp', QualityCheck(EXP, OBS, 'RXASP',TF_emp=True,delta=0.01))

    selected_covariates = ['AGE', 'SEX', 'RXASP', 'Y']
    EXP, OBS = HideCovarOBS(EXP, OBS,selected_covariates)
    # IST = HideCovarDF(IST)

    return [EXP, OBS]

def QualityCheck(EXP,OBS,X,TF_emp = False,delta=0.01):
    if TF_emp:
        LB, HB = EmpiricalComputeBound(OBS, X,delta,len(OBS))
    else:
        LB, HB = ComputeBound(OBS, X)
    U = GroundTruth(EXP, X)
    TF_Case2 = CheckCase2(HB, U)

    lx0, lx1 = LB
    hx0, hx1 = HB
    Ux0, Ux1 = U

    print([lx0, Ux0, hx0])
    print([lx1, Ux1, hx1])
    # print(TF_Case2)
    return(TF_Case2)

if __name__ == "__main__":
    EXP,OBS = RunGenData()
    #
    #
    # chosen_variables = ['SEX', 'AGE', 'RCONSC',
    #                     'RXASP', 'EXPDD'
    #                     ]
    # discrete_variables = ['SEX', 'RCONSC', 'RXASP']
    # continuous_variables = ['AGE', 'EXPDD']
    # necessary_set = ['RXASP', 'AGE', 'SEX', 'RCONSC']
    # params = [0.2, -0.2, 0.2, 0.2]
    #
    # # Data load
    # ## Preprocessing
    # IST = pd.read_csv('IST.csv')
    # IST = ReduceIST(IST, chosen_variables)
    # IST_orig = copy.copy(IST)
    # IST = IndexifyDisc(IST, discrete_variables)
    # IST['SEX'] = 1 - IST['SEX']
    # IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)
    #
    # IST = ContToDisc(IST, continuous_variables)
    # EXP = GenAddY(IST, necessary_set, params)
    # OBS = GenOBS(EXP, params)
    # print("OBS",ComputeEffect(OBS, 'RXASP', 'Y'))
    # print("EXP",ComputeEffect(EXP, 'RXASP', 'Y'))

    # EXP,OBS = RunGenData()
    # print(ComputeEffect(EXP,'RXASP','Y'))
    # print(ComputeEffect(OBS, 'RXASP', 'Y'))
    # print('Case 2 (no emp)',QualityCheck(EXP,OBS,'RXASP',TF_emp=False))
    # print('Case 2 (emp)', QualityCheck(EXP, OBS, 'RXASP', TF_emp=True))


