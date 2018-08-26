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

    # Exclude patients being dead due to other reasons (irrevant stroke)
    # ## Patients dead from other causes are exlucded.
    # IST = IST.loc[(IST['DEAD7'] == 0) & (IST['DEAD8'] == 0)]
    # chosen_variables = ['SEX', 'AGE',
    #                     'RSLEEP', 'RATRIAL', 'RCONSC', 'RDELAY', 'RVISINF', 'RSBP',
    #                     'RXASP','RXHEP',
    #                     'FRECOVER','FDEAD','DRSISC','DPE','H14','DSIDE','DALIVE',
    #                     'EXPDD','DDEAD'
    #                     ]
    # # Include only chosen variables (covariaets)
    #
    # # Delete row with missing variables.
    # IST = IST.dropna()
    # IST_orig = copy.copy(IST)
    # IST = IndexifyDisc(IST)
    #
    # # Generate continuous Y
    # # FRECOVER {0,1} (1 if recovered // 0 if dependent or dead)
    # # IST['EXPDD'] is a predicted probabililty of being dead+dependent.
    # # i.e., 1-IST['EXPDD'] is a prob of recovery.
    # ## outcome is higher if actually recovered despite low recovery prob.
    # outcome = IST['FRECOVER'] - (1 - IST['EXPDD'])
    # outcome = (outcome + 1) / 2 # Normalizing to [0,1]
    # outcome = pd.DataFrame({'Y': outcome})
    # IST = pd.concat([IST, outcome], axis=1)
    # IST_orig = pd.concat([IST_orig, outcome], axis=1)
    # return IST_orig

def IndexifyDisc(IST, discrete_variables):
    # discrete_variables = ['SEX', 'RSLEEP', 'RATRIAL', 'RCONSC',
    #                       'RVISINF', 'RXASP','RXHEP',
    #                       'FRECOVER','FDEAD','DRSISC','DPE','DSIDE','DALIVE','DDEAD'
    #                       ]

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

def GenOBS(EXP, seed_obs = 1):
    X = 'X'
    weight_sick = 0.1
    weight_treatment = 0.9
    sample_list = []
    for idx in range(len(EXP)-1):
        elem = EXP.iloc[idx]
        elem_EXPD = elem['EXPDD']
        elem_treat = elem[X]

        # MAKE THIS CODE MORE INTERPRETABLE
        if elem_treat == 0:
            if elem_EXPD < 0.7: # Healthy
                prob = 0.1
            else:
                prob = 0.9
        else:
            if elem_EXPD < 0.7:
                prob = 0.9
            else:
                prob = 0.1

        # Computing the selection probability of patients idx
        selection_prob = np.dot([weight_sick, weight_treatment],[prob, 1-elem_treat])
        print(idx, prob, selection_prob)

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
    selected_covariates = ['AGE', 'SEX','EXPDD', 'RXASP', 'Y']

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
    possible_case = 2**32 -1
    seed_num = np.random.randint(possible_case)
    np.random.seed(seed_num)

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



def GenAddY(df,necessary_set = ['RXASP','AGE','SEX','RCONC'],params = [0.6, -0.2, 0.2, 0.2]):
    df['Y'] = df[necessary_set[0]] * params[0]
    for idx in range(1,len(necessary_set)):
        df['Y'] = df['Y'] + df[necessary_set[idx]]*params[idx]
    df['Y'] = df['Y'] + np.random.normal(loc=0,scale=0.1,size=len(df))
    return df








def RunGenData(sample_N=12000, remember_seed = 1444260861):
    sample_N = 12000
    # remember_seed = 1444260861
    remember_seed = 2384686002

    # chosen_variables = ['SEX', 'AGE',
    #                     'RSLEEP', 'RATRIAL', 'RCONSC', 'RDELAY', 'RVISINF', 'RSBP',
    #                     'RXASP','RXHEP',
    #                     'FRECOVER','FDEAD','DRSISC','DPE','H14','DSIDE','DALIVE',
    #                     'EXPDD','DDEAD'
    #                     ]
    chosen_variables = ['SEX', 'AGE', 'RCONSC',
                        'RXASP', 'EXPDD'
                        ]
    discrete_variables = ['SEX','RCONSC','RXASP']
    continuous_variables = ['AGE','EXPDD']
    necessary_set = ['RXASP','AGE','SEX','RCONSC']
    params = [0.6,-0.2,0.2,0.2]

    # Data load
    ## Preprocessing
    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST,chosen_variables)
    IST_orig = copy.copy(IST)
    IST = IndexifyDisc(IST,discrete_variables)
    IST['SEX'] = 1-IST['SEX']
    IST['RCONSC'] = 1*(IST['RCONSC'] == 0) + 0*(IST['RCONSC'] == 2)  + 2*(IST['RCONSC']==1)

    IST = ContToDisc(IST,continuous_variables)
    IST = GenAddY(IST,necessary_set,params)

    # listOutcome = []
    # for idx in range(len(IST)):
    #     if idx % 100 == 0:
    #         print(idx)
    #     elem = IST.iloc[idx]
    #     if elem['RXASP'] == 0:
    #         outcome = elem['Y']*0.2
    #         listOutcome.append(outcome)
    #     else:
    #         outcome = elem['Y'] * 0.8
    #         listOutcome.append(outcome)
    # Z = pd.DataFrame(index=IST.index,data={'Z':listOutcome})
    # IST = pd.concat([IST,Z],axis=1)

    # Y = GenY(IST,IST_orig)
    # Y = np.array(Y)
    # Y = (Y+3)/7
    # Y = pd.DataFrame(index=IST.index,data={'Y':Y})
    # IST = pd.concat([IST,Y],axis=1)
    # EXP = GenEXP(IST, sample_N, remember_seed)  # SO FAR GODO
    #
    #
    # # IST_X = list()
    # # IST_sample = []
    # # for idx in range(len(IST)):
    # #     elem_IST = IST.iloc[idx]
    # #     if (IST['RXASP'].iloc[idx] == 1) and (IST['RXHEP'].iloc[idx]==1):
    # #         IST_X.append(1)
    # #         IST_sample.append(elem_IST)
    # #     elif (IST['RXASP'].iloc[idx] == 1) and (IST['RXHEP'].iloc[idx] == 2):
    # #         IST_X.append(0)
    # #         IST_sample.append(elem_IST)
    #
    # IST_sample = pd.DataFrame(IST_sample)
    # IST_X = pd.DataFrame({'X':IST_X})
    # IST = pd.concat([IST_sample,IST_X],axis=1)
    # EXP = copy.copy(IST)

    # remember_seed = SeedFinding(IST,sample_N=12000, alpha=0.01)

    # tempOBS = GenOBS(EXP)
    # print(ComputeEffect(EXP,'Y'))
    # print(ComputeEffect(tempOBS, 'Y'))
    # print(QualityCheck(EXP,tempOBS,'RXASP'))
    OBS = GenOBS(EXP,20)

    EXP, OBS = HideCovarOBS(EXP, OBS)
    IST = HideCovarDF(IST)

    return [IST, EXP, OBS]

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
    # print(TF_Case2)
    return(TF_Case2)

if __name__ == "__main__":
    IST,EXP,OBS = RunGenData()

    # import pandas as pd
    # import scipy.io
    #
    # scipy.io.savemat('OBS.mat', {'OBS': OBS.to_dict("list")})
    # scipy.io.savemat('EXP.mat', {'EXP': EXP.to_dict("list")})


    # nOBS = 10000
    # sumLB = np.array([0,0])
    # sumHB = np.array([0,0])
    # for idx in range(100):
    #     LB,HB = EmpiricalComputeBound(OBS.sample(n=nOBS), 'RXASP', 0.01)
    #     LB = np.array(LB)
    #     HB = np.array(HB)
    #     sumLB = sumLB + LB
    #     sumHB = sumHB + HB
    # print(sumLB/100,sumHB/100)

    # for age in [0, 1]:
    #     for sex in [0, 1]:
    #         for biexpd in [0, 1]:
    #             for x in [0, 1]:
    #                 expected = np.mean(EXP[(EXP['AGE'] == age) & (EXP['SEX'] == sex) & (EXP['BiEXPD'] == biexpd) & (
    #                 EXP['RXASP'] == x)]['Y'])
    #                 print('E[Yx|u]', expected, 'when X=', x, 'Z=', '(', age, sex, biexpd, ')')


    listSample = []
    IST_sample = IST[['RCONSC', 'AGE', 'SEX', 'RXASP', 'EXPDD']]
    # IST_sample2 = copy.copy(IST_sample)
    for idx in range(len(IST_sample)):
        if idx % 50 == 0:
            print(idx)
        elem = IST_sample.iloc[idx]
        if elem['SEX'] == 'F':
            elem['SEX'] = 1
        else:
            elem['SEX'] = 0

        if elem['RXASP'] == 'Y':
            elem['RXASP'] = 1
        elif elem['RXASP'] == 'N':
            elem['RXASP'] = 0
        else:
            continue

        if elem['RCONSC'] == 'F':
            elem['RCONSC'] = 2
        elif elem['RCONSC'] == 'D':
            elem['RCONSC'] = 1
        elif elem['RCONSC'] == 'U':
            elem['RCONSC'] = 0
        else:
            continue
        listSample.append(elem)
    ISTSample = pd.DataFrame(listSample)
    import scipy.io

    scipy.io.savemat('ISTSample.mat', {'ISTSample': ISTSample.to_dict("list")})




