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

def drawArmFromPolicy(pl,Z):
    prob0, prob1 = pl(Z[0],Z[1])
    return np.random.binomial(n=1, p=prob1, size=1)[0]

def SigTestPl(df1, df2):
    pval = ttest_ind(df1['Y'],df2['Y'],equal_var=False).pvalue
    return pval

def ExpectedOutcomePl(EXP,pl):
    X = 'RXASP'
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

def ComputeXYEffect(df,X,Y):
    return [np.mean(df[df[X]==0][Y]),np.mean(df[df[X]==1][Y])]

def GenOBS(EXP, seed_obs = 1):
    pxu = [0, 0, 0, 0, 0, 0, 1, 0.2, 0.1, 0, 0, 0]
    listSample = []
    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_treat = elem['RXASP']
        elem_sex = elem['SEX']
        elem_age = elem['AGE']
        elem_RCONSC = elem['RCONSC']

        u = int(6*elem_age + 3*elem_sex + elem_RCONSC + 1)
        print(u)
        x = np.random.binomial(1,pxu[u-1])
        if x == elem_treat:
            listSample.append(elem)
    OBS = pd.DataFrame(listSample)
    return OBS



    # np.random.seed(seed_obs)
    # sample_list = []
    #
    # for idx in range(len(EXP)):
    #     elem = EXP.iloc[idx]
    #
    #
    #     elem_treat = elem['RXASP']
    #     elem_age = elem['AGE']
    #     elem_sex = elem['SEX']
    #
    #     if elem_age == 0 and elem_sex == 0 and elem_treat == 1:
    #         prob = 0.9955
    #     else:
    #         prob = 0.0045
    #
    #     selection_prob = prob
    #     if np.random.binomial(1, selection_prob) == 0:
    #         continue
    #     else:
    #         sample_list.append(elem)
    #
    # OBS = pd.DataFrame(sample_list)
    # return OBS


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
    pl2 = lambda age, sex: [low_prob, high_prob] if ((age == 0) and (sex == 1)) else [high_prob, low_prob]
    pl3 = lambda age, sex: [low_prob, high_prob] if ((age == 1) and (sex == 0)) else [high_prob, low_prob]
    pl4 = lambda age, sex: [low_prob, high_prob] if ((age == 1) and (sex == 1)) else [high_prob, low_prob]
    policy_list = [pl1, pl2, pl3, pl4]
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

def GenAddY(df,necessary_set):
    df['Y'] = df[necessary_set[0]] + 0.5*df[necessary_set[2]] + 0.2*(df[necessary_set[3]]-1) - df[necessary_set[0]]*df[necessary_set[1]]
    df['Y'] = df['Y'] + np.random.normal(loc=0,scale=0.1,size=len(df))
    df['Y'] = 1 / (1 + np.exp(-df['Y']))
    return df

def GenEXPY(EXP,pl):
    np.random.seed(1)
    listY = []
    X = 'RXASP'
    m = 0
    n = 0

    num_iter = 3000

    for idx in range(num_iter):
        # if idx % 100 == 0:
        #     print(idx)
        z = list(EXP[['AGE','SEX']].iloc[idx])
        age,sex = z
        x = drawArmFromPolicy(pl,z)
        y = list(EXP[(EXP['AGE']==age) & (EXP['SEX']==sex) & (EXP[X]==x)].sample(1)['Y'])[0]

        n += 1
        m = ((n-1)*m + y)/n

        # listY.append(y)
    return m



def RunGenData():
    chosen_variables = ['SEX', 'AGE', 'RCONSC',
                        'RXASP', 'EXPDD'
                        ]
    discrete_variables = ['SEX', 'RCONSC', 'RXASP']
    continuous_variables = ['AGE', 'EXPDD']
    necessary_set = ['RXASP', 'AGE', 'SEX', 'RCONSC']

    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST, chosen_variables)
    IST_orig = copy.copy(IST)
    IST = IndexifyDisc(IST, discrete_variables)
    IST['SEX'] = 1 - IST['SEX']
    IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)

    IST = ContToDisc(IST, continuous_variables)
    EXP = GenAddY(IST, necessary_set)


    # Define policies
    low_prob = 0.005
    high_prob = 0.995
    listPolicy = PolicyGen(low_prob, high_prob)
    listY = []

    for pl in listPolicy:
        y_pl = GenEXPY(EXP,pl)
        listY.append(y_pl)

    OBS = GenOBS(EXP)

    # EXP1 = HideCovarOBS(EXP1)
    # EXP2 = HideCovarOBS(EXP2)
    # OBS = HideCovarOBS(OBS)
    # listEXP = [EXP1,EXP2]

    return [EXP,OBS]

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

if __name__ == "__main__":
    EXP,OBS = RunGenData()


#
# listEXP, OBS = RunGenData()
# policy_list = PolicyGen()
# QualityCheck(listEXP,OBS,policy_list)
