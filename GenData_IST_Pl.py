import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_ind
from sklearn import preprocessing

def HideCovarOBS(df):
    selected_covariates = ['AGE', 'SEX', 'RXASP', 'Y']

    ## Resulting dataset
    df = df[selected_covariates]
    return df

def IST_LabelEncoder(IST, ColName):
    le = preprocessing.LabelEncoder()

    COL = copy.copy(IST[ColName])
    list_label = list(pd.unique(COL))
    le.fit(list_label)

    COL = le.transform(COL)

    IST[ColName] = COL
    return IST

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
    # IST['AGE'] = ThreeCategorize(IST['AGE'])
    IST['RSBP'] = ThreeCategorize(IST['RSBP'])
    IST['RDELAY'] = ThreeCategorize(IST['RDELAY'])
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

def drawPolicy(pl,Z):
    prob0, prob1 = pl(Z[0],Z[1])
    return np.random.binomial(n=1, p=prob1, size=1)[0]

def SigTestPl(df1, df2):
    pval = ttest_ind(df1['Y'],df2['Y'],equal_var=False).pvalue
    return pval

def ExpectedOutcomePl(EXP,pl):
    sum_prob = 0
    for age in [0,1]:
        for sex in [0,1]:
            Pz = len(EXP[(EXP['AGE']==age)&(EXP['SEX']==sex)])/len(EXP)
            probs = pl(age,sex)
            Yz = EXP[(EXP['AGE'] == age) & (EXP['SEX'] == sex)]
            for x in [0,1]:
                Yx_z = Yz[Yz[X]==x]['Y']
                EYx_z = np.mean(Yx_z)
                sum_prob_val = EYx_z * probs[x] * Pz
                # print(sum_prob_val, EYx_z, probs[x],Pz,age,sex)
                sum_prob += sum_prob_val
    return sum_prob

# Gen EXP
def GenEXPPl(IST,policy_list, seed_num=123):
    np.random.seed(seed_num)
    listSamplePlIST = []
    X = 'RXASP'
    for idx in range(len(policy_list)):
        listSamplePlIST.append([])

    for idx in range(len(IST)):
        elemIST = IST.iloc[idx]
        z = list(elemIST[['AGE', 'SEX']])
        x = int(elemIST[X])
        for plidx in range(len(policy_list)):
            pl = policy_list[plidx]
            probs = pl(z[0],z[1])
            sampling_prob = probs[x]
            if np.random.binomial(1,sampling_prob) == 1:
                listSamplePlIST[plidx].append(elemIST)

    for plidx in range(len(policy_list)):
        listSamplePlIST[plidx] = pd.DataFrame(listSamplePlIST[plidx])
    return listSamplePlIST

def GenOBS(EXP, seed_obs = 123):
    np.random.seed(seed_obs)
    weight_sick = 0.99
    weight_treatment = 0.01

    sample_list = []

    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_EXPD = elem['EXPDD']
        elem_treat = elem['RXASP']
        elem_age = elem['AGE']
        elem_sex = elem['SEX']

        if elem_age == 0 and elem_sex == 0 and elem_treat == 1:
            prob = 0.9955
        else:
            prob = 0.0045

        selection_prob = prob
        if np.random.binomial(1, selection_prob) == 0:
            continue
        else:
            sample_list.append(elem)

    OBS = pd.DataFrame(sample_list)
    return OBS


def GenOBSPl(EXP,sample_N, seed_num):
    np.random.seed(seed_num)
    return EXP.sample(sample_N)

def BoundsPl(OBS,pl):
    X = 'RXASP'
    sum_prob_lb = 0
    for age in [0, 1]:
        for sex in [0, 1]:
            probs = pl(age, sex)
            for x in [0,1]:
                P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X]==x)]) / len(OBS)
                pi_xz = probs[x]
                EY_xz = np.mean(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X]==x)]['Y'])
                sum_prob_lb += EY_xz * pi_xz * P_xz

    sum_prob_ub = 0
    LB = copy.copy(sum_prob_lb)
    for age in [0, 1]:
        for sex in [0, 1]:
            probs = pl(age, sex)
            for x in [0,1]:
                P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
                non_pi_xz = probs[1-x]
                sum_prob_ub += P_xz * non_pi_xz
    UB = LB + sum_prob_ub
    return [LB,UB]

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

def PolicyGen(low_prob=0.005, high_prob=0.995):
    pl1 = lambda age, sex: [low_prob, high_prob] if ((age == 0) and (sex == 0)) else [high_prob, low_prob]
    pl2 = lambda age, sex: [high_prob, low_prob] if ((age == 0) and (sex == 0)) else [low_prob, high_prob]
    policy_list = [pl1, pl2]
    return policy_list

def LB_U_HB(listEXP,OBS, policy_list):
    EXP_pl1, EXP_pl2 = listEXP
    pl1, pl2 = policy_list

    EY1 = np.mean(EXP_pl1['Y'])
    EY2 = np.mean(EXP_pl2['Y'])

    LB1, HB1 = BoundsPl(OBS, pl1)
    LB2, HB2 = BoundsPl(OBS, pl2)

    U = [EY1, EY2]
    LB = [LB1,LB2]
    HB = [HB1, HB2]

    return [LB,U,HB]


def RunGenData():
    # Load dataset
    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST)
    IST = IndexifyDisc(IST)
    IST = ContToDisc(IST)

    # Define policies
    low_prob = 0.005
    high_prob = 0.995
    policy_list = PolicyGen(low_prob, high_prob)

    listEXP = GenEXPPl(IST, policy_list)
    EXP1, EXP2 = listEXP
    # print(np.mean(EXP_pl1['Y']), ExpectedOutcomePl(EXP_pl1,pl1))
    # print(np.mean(EXP_pl2['Y']), ExpectedOutcomePl(EXP_pl2,pl2))

    OBS = GenOBS(IST)

    EXP1 = HideCovarOBS(EXP1)
    EXP2 = HideCovarOBS(EXP2)
    OBS = HideCovarOBS(OBS)
    listEXP = [EXP1,EXP2]

    return [listEXP, OBS, IST]

def QualityCheck(listEXP, OBS, policy_list):
    LB,U,HB = LB_U_HB(listEXP,OBS,policy_list)
    LB1,LB2 = LB
    U1, U2 = U
    HB1, HB2 = HB

    TF_case2 = CheckCase2(HB,U)
    print(LB1, U1, HB1)
    print(LB2, U2, HB2)
    print('CASE 2',TF_case2)
    return [LB,U,HB]

#
# listEXP, OBS = RunGenData()
# policy_list = PolicyGen()
# QualityCheck(listEXP,OBS,policy_list)
