import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_ind
from sklearn import preprocessing
import pickle

def HideCovarDF(DF,selected_covariates):
    ## Resulting dataset
    DF = DF[selected_covariates]
    return DF

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
    pxu = [0, 0, 0, 0, 0, 0, 1, 0.2, 0.1, 0.1, 0, 0]
    listSample = []
    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_treat = elem['RXASP']
        elem_sex = elem['SEX']
        elem_age = elem['AGE']
        elem_RCONSC = elem['RCONSC']

        u = int(6*elem_age + 3*elem_sex + elem_RCONSC + 1)
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

def BoundsPl(OBS, OBS_Z, OBS_X,pl, zName, xName):
    zDomain = np.unique(OBS_Z)
    xDomain = np.unique(OBS_X)
    N = len(OBS)

    sum_prob_lb = 0
    zidx = 0
    for z in zDomain:
        z = int(z)
        pz = listPz[zidx]
        zidx += 1
        for x in xDomain:
            x = int(x)
            pl_xz = FunEval(pl,z)
            print(pl_xz, x)
            pl_xz = pl_xz[x]
            EY_xz = np.mean(OBS[(OBS[zName]==z) & (OBS[xName]==x)]['Y'])
            if pd.isnull(EY_xz):
                EY_xz = 0
            sum_prob_lb += EY_xz * pl_xz * pz

    sum_prob_ub = 0
    LB = copy.copy(max(sum_prob_lb, 0))

    zidx = 0
    for z in zDomain:
        pz = listPz[zidx]
        zidx += 1
        for x in xDomain:
            x = int(x)
            pxz = len(OBS[(OBS[zName]==z) & (OBS[xName]==x)]) / N
            if pd.isnull(pxz):
                pxz = 0
            pi_nonx_z = FunEval(pl,z)
            pi_nonx_z = pi_nonx_z[1-x]
            sum_prob_ub += pxz * pi_nonx_z
    HB = LB + sum_prob_ub
    HB = min(HB, 1)
    return [LB, HB]


def EmpBoundsPl(OBS,zName,listPolicy,delta,N):
    delta = delta / len(listPolicy)
    fn = np.sqrt(((2 * N) ** (-1)) * (np.log(4) - np.log(delta)))

    Ys = list(OBS['Y'])
    Xs = list(OBS['RXASP'])
    Zs = list(OBS[zName])

    listLB = []
    listHB = []

    for plidx in range(len(listPolicy)):
        sumval = 0
        pl = listPolicy[plidx]
        for idx in range(N):
            y = Ys[idx]
            x = int(Xs[idx])
            z = Zs[idx]
            pl_xz = FunEval(pl, z)
            pl_xz = pl_xz[x]
            sumval += (y*pl_xz)
        LB_orig = sumval/N
        LB = max(0,LB_orig - fn)

        sumval = 0
        for idx in range(N):
            x = int(Xs[idx])
            z = Zs[idx]
            pl_xz = FunEval(pl, z)
            pl_xz = pl_xz[1-x]
            sumval += pl_xz
        HB_orig = LB_orig + sumval/N
        HB = HB_orig + fn
        HB = min(HB,1)

        listLB.append(LB)
        listHB.append(HB)

    return [listLB,listHB]



def CheckCase2(h_subopt, u_opt):
    if h_subopt < u_opt:
        return True
    else:
        return False



def PolicyGen(low_prob, high_prob):
    pl1 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 0) and (z1 == 0)) else [high_prob, low_prob]
    pl2 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 0) and (z1 == 1)) else [high_prob, low_prob]
    pl3 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 0) and (z1 == 2)) else [high_prob, low_prob]
    pl4 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 1) and (z1 == 0)) else [high_prob, low_prob]
    pl5 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 1) and (z1 == 1)) else [high_prob, low_prob]
    pl6 = lambda z0, z1: [low_prob, high_prob] if ((z0 == 1) and (z1 == 2)) else [high_prob, low_prob]

    policy_list = [pl1, pl2, pl3, pl4, pl5, pl6]
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

# def GenEXPY(EXP,pl,covariate_Z):
#     np.random.seed(1)
#     listY = []
#     X = 'RXASP'
#     m = 0
#     n = 0
#
#     num_iter = 5000
#
#     for idx in range(num_iter):
#         # if idx % 100 == 0:
#         #     print(idx)
#         z = list(EXP[covariate_Z].iloc[idx])
#         z0,z1= z
#         x = drawArmFromPolicy(pl,z)
#         y = list(EXP[(EXP[covariate_Z[0]]==z0) & (EXP[covariate_Z[1]]==z1) & (EXP[X]==x)].sample(1)['Y'])[0]
#
#         n += 1
#         m = ((n-1)*m + y)/n
#
#         # listY.append(y)
#     return m

def ComputePz(EXP_Z, zName, EXP):
    # Assume z 1D
    zDomain = np.unique(EXP_Z)
    N = len(EXP)
    listPz = []
    for z in zDomain:
        pz = len(EXP[EXP[zName]==z])/N
        listPz.append(pz)
    return listPz

def RewardFun(x, s, r, a):
    y = x + 0.5 * s + 0.2 * (r - 1) - x * a + np.random.normal(0,1)
    y = 1 / (1 + np.exp(-y))
    return y

def FunEval(pl,z):
    pl_1z = pl(z)
    return [1-pl_1z, pl_1z]

def RunGenData():
    chosen_variables = ['SEX', 'AGE', 'RCONSC',
                        'RXASP',
                        ]
    discrete_variables = ['SEX', 'RCONSC', 'RXASP']
    continuous_variables = ['AGE']
    necessary_set = ['RXASP', 'AGE', 'SEX', 'RCONSC']
    coviarateZ = ['SEX','RCONSC']

    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST, chosen_variables)
    IST_orig = copy.copy(IST)
    IST = IndexifyDisc(IST, discrete_variables)
    IST['SEX'] = 1 - IST['SEX']
    IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)

    EXP = ContToDisc(IST, continuous_variables)
    EXP = GenAddY(EXP,necessary_set)
    ##### GEN Y
    OBS = GenOBS(EXP)

    EXP_SEX = list(EXP['SEX'])
    EXP_AGE = list(EXP['AGE'])
    EXP_RCONSC = list(EXP['RCONSC'])
    EXP_RXASP = list(EXP['RXASP'])

    OBS_SEX = list(OBS['SEX'])
    OBS_AGE = list(OBS['AGE'])
    OBS_RCONSC = list(OBS['RCONSC'])
    OBS_RXASP = list(OBS['RXASP'])


    # Define policies
    pl1 = lambda z: 0.01 if z == 0 else 0.02
    pl2 = lambda z: 0.05 if z == 0 else 0.1
    pl3 = lambda z: 0.97 if z == 0 else 0.99
    pl4 = lambda z: 0.1 if z == 0 else 0.05
    listPolicy = [pl1, pl2, pl3, pl4]


    # low_prob = 0.01
    # high_prob = 0.99
    # listPolicy = PolicyGen(low_prob, high_prob)

    listU = []
    for pl in listPolicy:
        m = 0
        n = 0
        for s,r,a in zip(EXP_SEX, EXP_RCONSC, EXP_AGE):
            x = np.random.binomial(1,FunEval(pl,s)[1])
            y = RewardFun(x,s,r,a)
            n += 1
            m = ((n - 1) * m + y) / n
        listU.append(m)

    listPz = ComputePz(OBS_SEX,'SEX',OBS)
    listLB, listHB = EmpBoundsPl(OBS, 'SEX', listPolicy, 0.01, len(OBS))

    pickle.dump(EXP, open('sim_instance/EXP.pkl','wb'))
    pickle.dump(OBS, open('sim_instance/OBS.pkl', 'wb'))
    # pickle.dump(listPolicy, open('sim_instance/listPolicy.pkl', 'wb'))
    pickle.dump(listU, open('sim_instance/listU.pkl', 'wb'))
    pickle.dump(listPz, open('sim_instance/listPz.pkl', 'wb'))
    pickle.dump(listLB, open('sim_instance/listLB.pkl','wb'))
    pickle.dump(listHB, open('sim_instance/listHB.pkl','wb'))

    return(EXP,OBS,listPolicy,listU)










    # EXP1 = HideCovarOBS(EXP1)
    # EXP2 = HideCovarOBS(EXP2)
    # OBS = HideCovarOBS(OBS)
    # listEXP = [EXP1,EXP2]



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
    EXP, OBS, listU, listPolicy = RunGenData()
    listPz = ComputePz(EXP)
    optpl = np.argmax(listU)
    uopt = listU[optpl]

    obs_covar = ['SEX','RCONSC','RXASP','Y']
    covZ = ['SEX','RCONSC']
    EXP_orig = copy.copy(EXP)
    EXP = HideCovarDF(EXP,obs_covar)
    OBS = HideCovarDF(OBS,obs_covar)

    listBdd = []
    listHB = []
    listLB = []

    for pl in listPolicy:
        LB,HB = BoundsPl(OBS,listPz,pl,X='RXASP',covZ=['SEX','RCONSC'])
        listBdd.append([LB,HB])
        listHB.append(HB)
        listLB.append(LB)


    delta = 0.01
    listLBe, listHBe = EmpBoundsPl(OBS,covZ,listPolicy,delta,len(OBS))

    pickle.dump(EXP, open('EXP.pkl','wb'))
    pickle.dump(OBS, open('OBS.pkl', 'wb'))
    pickle.dump(listU, open('listU_005095.pkl', 'wb'))
    # pickle.dump(listPolicy, open('listPolicy0109.pkl', 'wb'))

    ''' Check Case 2 '''
    # for plidx in range(len(listPolicy)):
    #     print(CheckCase2(listHB[plidx],uopt))

    # for pldx in range(len(listPolicy)):
    #     print(listBdd[pldx], listU[pldx])





#
# listEXP, OBS = RunGenData()
# policy_list = PolicyGen()
# QualityCheck(listEXP,OBS,policy_list)
