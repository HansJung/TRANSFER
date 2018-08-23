import pandas as pd
import numpy as np
import GenData_IST_Pl as GenData
import itertools
import matplotlib.pyplot as plt
import scipy.io

def ObserveZSim(Pz):
    zPossible = [[0, 0], [0, 1], [1, 0], [1, 1]]
    zCase = np.random.choice(len(Pz), p=Pz)
    return zPossible[zCase]


def ExpectedOutcomeSim(pl):
    zPossible = [[0, 0], [0, 1], [1, 0], [1, 1]]
    sumval = 0
    for idx in range(len(Pz)):
        z = zPossible[idx]
        pz = Pz[idx]
        for x in [0, 1]:
            pi_xz = pl(z[0], z[1])[x]
            expected_xz = EYx_z[idx][x]
            sumval += pz * pi_xz * expected_xz
    return sumval

def ComputePxz(OBS,age,sex,x):
    X = 'RXASP'
    P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
    return P_xz

def ComputePz(OBS,age,sex):
    Pz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex)]) / len(OBS)
    return Pz

def ComputePzSim(Pz,age,sex):
    if age == 0 & sex == 0:
        return Pz[0]
    elif age == 0 & sex == 1:
        return Pz[1]
    elif age == 1 & sex == 0:
        return Pz[2]
    else:
        return Pz[3]


def ReceiveRewardsSim(plidxChosen,armChosen,listU):
    ua = listU[armChosen]
    reward = np.random.binomial(1,ua)
    return reward

def ReceiveRewards(plidxChosen, armChosen, listEXP):
    X = 'RXASP'
    EXPi = listEXP[plidxChosen]
    reward = list(EXPi[EXPi[X] == armChosen].sample(1)['Y'])[0]

    # EXPi[X == armChosen]
    # reward = EXPi.iloc[dictlistNumPolicyArm[plidxChosen][armChosen]]['Y']
    return reward

def ComputeMpqSim(p,q):
    def f1(x):
        return x*np.exp(x-1)-1

    sumProb = 0
    for age in [0,1]:
        for sex in [0,1]:
            pofz = ComputePzSim(Pz,age,sex)
            for x in [0,1]:
                pxz = p(age,sex)[x]
                qxz = q(age,sex)[x]
                sumProb += f1(pxz/qxz)*qxz*pofz

    return (1+np.log(1+sumProb))

def ComputeMpq(p,q,OBS):
    def f1(x):
        return x*np.exp(x-1)-1

    sumProb = 0
    for age in [0,1]:
        for sex in [0,1]:
            Pz = ComputePz(OBS,age,sex)
            for x in [0,1]:
                pxz = p(age,sex)[x]
                qxz = q(age,sex)[x]
                sumProb += f1(pxz/qxz)*qxz*Pz

    return (1+np.log(1+sumProb))

def ComputeMatMSim(listPolicy):
    N_poly = len(listPolicy)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        pk = listPolicy[k]
        pj = listPolicy[j]
        M_mat[k,j] = ComputeMpqSim(pk,pj)
    return M_mat

def ComputeMatM(listPolicy,OBS):
    N_poly = len(listPolicy)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        pk = listPolicy[k]
        pj = listPolicy[j]
        M_mat[k,j] = ComputeMpq(pk,pj,OBS)
    return M_mat

def ObserveZ(IST,t):
    return list(IST.sample(1)[['AGE','SEX']])
    # return list(IST.iloc[t-1][['AGE', 'SEX']])

def ObserveZSim(Pz):
    zPossible = [[0,0],[0,1],[1,0],[1,1]]
    zCase = np.random.choice(len(Pz), p=Pz)
    return zPossible[zCase]

def ComputeZkt(dictNumPolicy,nPolicy, M, k):
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        sumval += dictNumPolicy[j] / Mkj
    return sumval

def ComputeDynamicMu(listPolicy, dictLastElem):
    eps = lambda t: 2 / (t ** 2)
    block = lambda t: 2 * np.log(2 / eps(t))

def ComputeVal(k, j, xjs,yjs,zjs,listPolicy,M,t):
    eps = lambda t: 2 / (t ** 2)
    block = lambda t: 2 * np.log(2 / eps(t))

    Mkj = M[k, j]
    blockval = block(t) * Mkj

    pik = listPolicy[k]
    pij = listPolicy[j]

    invval = (pik(zjs[0], zjs[1])[xjs] / pij(zjs[0], zjs[1])[xjs])
    if invval <= blockval:
        result = (1 / Mkj) * yjs * invval
    else:
        result = 0
    return result



def ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, M,k,t):
    eps = lambda t: 2/(t**2)
    block = lambda t: 2*np.log(2/eps(t))

    nPolicy = len(listPolicy)
    Zkt = ComputeZkt(dictNumPolicy,nPolicy,M,k)
    pik = listPolicy[k]
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        blockval = block(t)*Mkj
        pij = listPolicy[j]
        Xj = dictdictlistPolicyData[j]['X']
        Yj = dictdictlistPolicyData[j]['Y']
        Zj = dictdictlistPolicyData[j]['Z']
        for s in range(len(Xj)):
            xjs = Xj[s]
            yjs = Yj[s]
            zjs = Zj[s]
            invval = (pik(zjs[0], zjs[1])[xjs]/pij(zjs[0],zjs[1])[xjs])
            if invval <= blockval:
                sumval += (1/Mkj)*yjs*invval
            else:
                sumval += 0
    return sumval/Zkt

def ComputeSk(dictNumPolicy, nPolicy, M, k, t):
    c1 = 16
    eps = 1e-8
    Zkt = ComputeZkt(dictNumPolicy, nPolicy, M, k)
    return 1.5*((np.sqrt(c1*t*np.log(t)) / Zkt)**(1/(1+eps)))

def PullArmFromPl(pl,z):
    probs = pl(z[0],z[1])
    return np.random.binomial(1,probs[1])

# def UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm,dictdictlistPolicyData,plidxChosen,armChosen,z,rewardReceived):
#     dictNumPolicy[plidxChosen] += 1
#     dictlistNumPolicyArm[plidxChosen][armChosen] += 1
#     dictdictlistPolicyData[plidxChosen]['Z'].append(z)
#     dictdictlistPolicyData[plidxChosen]['X'].append(armChosen)
#     dictdictlistPolicyData[plidxChosen]['Y'].append(rewardReceived)
#     return [dictNumPolicy,dictlistNumPolicyArm,dictdictlistPolicyData]

def UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm,dictdictlistPolicyData,plidxChosen,armChosen,z,rewardReceived):
    dictNumPolicy[plidxChosen] += 1
    dictlistNumPolicyArm[plidxChosen][armChosen] += 1
    return [dictNumPolicy,dictlistNumPolicyArm,dictdictlistPolicyData]

def UpdateOptProbAfterArm(dictNumPolicy,optpl,t):
    probOptPlChoose = dictNumPolicy[optpl]/t
    return probOptPlChoose

def RunDUCB(numRound,TF_causal, TF_sim):
    ''' Variable declaration '''
    dictNumPolicy = dict() # Number of choosing policy p
    dictLastElem = dict()
    dictlistNumPolicyArm = dict()
    dictdictlistPolicyData = dict()

    nPolicy = len(listPolicy)
    MatSumVal = np.zeros((nPolicy,nPolicy))
    dictMu = dict()
    dictUpper = dict()
    dictSk = dict()

    listProbOpt = []
    listTFArmCorrect = list()

    cummRegret = 0
    listCummRegret = []

    for plidx in range(nPolicy):
        dictNumPolicy[plidx] = 0
        dictLastElem[plidx] = 0
        dictlistNumPolicyArm[plidx] = [0, 0]
        dictdictlistPolicyData[plidx] = dict()
        dictdictlistPolicyData[plidx]['X'] = []
        dictdictlistPolicyData[plidx]['Y'] = []
        dictdictlistPolicyData[plidx]['Z'] = []

        dictMu[plidx] = 0
        dictSk[plidx] = 0
        dictUpper[plidx] = 0

    ''' Before start'''
    # Compute the M matrix
    if TF_sim == True:
        Mmat = ComputeMatMSim(listPolicy)
    else:
        Mmat = ComputeMatM(listPolicy, OBS)

    ''' Initial pulling by random choice'''
    t = 1
    # Observe Z
    if TF_sim == True:
        z = ObserveZSim(Pz)
    else:
        z = ObserveZ(IST, t)
    # Choose random expert
    plidxChosen = np.random.binomial(1, 0.5) # this is initial j
    pl = listPolicy[plidxChosen] # randomly chosen policy
    # Play an arm from pl
    armChosen = PullArmFromPl(pl, z)
    # Receive rewards
    if TF_sim == True:
        reward = ReceiveRewardsSim(plidxChosen,armChosen,U) # receive the reward
    else:
        reward = ReceiveRewards(plidxChosen, armChosen, listEXP)

    # Update information
    ## Update
    dictNumPolicy[plidxChosen] += 1
    # dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
    #     UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, z, reward)
    for k in range(nPolicy):
        # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
        MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen,yjs=reward,zjs=z,k=k,j=plidxChosen,listPolicy=listPolicy,M=Mmat,t=t)

    for k in range(nPolicy):
        dictMu[k] = 0
        Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
        for j in range(nPolicy):
            dictMu[k] += MatSumVal[k][j]
        dictMu[k] /= Zkt
        dictSk[k] = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
        dictUpper[k] = dictMu[k] + dictSk[k]

    print(t, dictMu)

    ''' Play! '''
    for t in range(2, numRound+2):
        # Observe Z
        if TF_sim == True:
            z = ObserveZSim(Pz)
        else:
            z = ObserveZ(IST, t)

        # Choose policy
        if TF_causal == True:
            listUpper = list(dictUpper.values())
            listU = list()
            for k in range(nPolicy):
                listU.append(np.max([np.min([listUpper[k],HB[k]]),LB[k]]))
            plidxChosen = np.argmax( listU )
        else:
            plidxChosen = np.argmax(list(dictUpper.values()))

        # listU = []
        # listmuk = []
        # listsk = []
        # for k in range(nPolicy):
        #
        #     muk = 0
        #     Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
        #     # for j in range(nPolicy):
        #     #     muk += dictLastElem[j]
        #     # muk = muk/Zkt
        #
        #     muk = ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, Mmat, k, t)
        #     listmuk.append(muk)
        #     sk = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
        #     listsk.append(sk)
        #     Uk = muk + sk
        #     if TF_causal:
        #         Uk = np.max([np.min([Uk, HB[k]]),LB[k]])
        #     listU.append(Uk)
        # if t % 1000 == 0:
        #     print(t, listmuk, listsk)
        #

        # # Choose Policy
        # plidxChosen = np.argmax(listU)
        pl = listPolicy[plidxChosen]
        # Play an arm
        armChosen = PullArmFromPl(pl, z)
        # Receive rewards
        if TF_sim == True:
            reward = ReceiveRewardsSim(plidxChosen, armChosen, U)
        else:
            reward = ReceiveRewards(plidxChosen, armChosen, listEXP)

        # Update information
        dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
            UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, z,
                           reward)
        ## Update
        for k in range(nPolicy):
            # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen, yjs=reward, zjs=z, k=k, j=plidxChosen,
                                                    listPolicy=listPolicy, M=Mmat, t=t)

        for k in range(nPolicy):
            dictMu[k] = 0
            Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            for j in range(nPolicy):
                dictMu[k] += MatSumVal[k][j]
            dictMu[k] /= Zkt
            dictSk[k] = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
            dictUpper[k] = dictMu[k] + dictSk[k]

        probOptPlChoose = UpdateOptProbAfterArm(dictNumPolicy, optpl, t)
        listProbOpt.append(probOptPlChoose)
        cummRegret += uopt - U[plidxChosen]
        listCummRegret.append(cummRegret)
        if plidxChosen== optpl:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        if t%1000 == 0:
            print(TF_causal,t,dictMu)
        # print(MatSumVal)
    return [listTFArmCorrect, listCummRegret]

def RunSimulation(numSim, numRound, TF_causal,TF_sim):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunDUCB(numRound,TF_causal,TF_sim)
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    return [MeanTFArmCorrect, MeanCummRegret]

TF_sim = True
TF_SaveResult = False
TF_plot = True
numRound = 2000
numSim = 200

if TF_sim == True:
    ''' Simulation instance '''
    Pz = [0.1,0.2,0.3,0.4]
    zPossible = [[0,0],[0,1],[1,0],[1,1]]
    EYx_z = [[0.1,0.5],[0.3,0.01],[0.36,0.66],[0.27,0.72]]

    LB = [0.1,0.3]
    HB = [0.4,0.7]
    U = list()
    listPolicy = GenData.PolicyGen()

    for pl in listPolicy:
        U.append(ExpectedOutcomeSim(pl))
    print(GenData.CheckCase2(HB,U))

else:
    ''' Real data instance '''
    listEXP, OBS, IST = GenData.RunGenData()
    listPolicy = GenData.PolicyGen()
    LB, U, HB = GenData.QualityCheck(listEXP, OBS, listPolicy)
    print(GenData.CheckCase2(HB, U))

optpl = np.argmax(U)
uopt = U[optpl]
usubopt = U[1-optpl]

MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim, numRound, TF_causal=False,TF_sim=TF_sim)
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim, numRound, TF_causal=True,TF_sim=TF_sim)

if TF_SaveResult:
    scipy.io.savemat('MeanTFArmCorrect.mat', mdict={'MeanTFArmCorrect': MeanTFArmCorrect})
    scipy.io.savemat('MeanCummRegret.mat', mdict={'MeanCummRegret': MeanCummRegret})
    scipy.io.savemat('MeanTFArmCorrect_C.mat', mdict={'MeanTFArmCorrect_C': MeanTFArmCorrect_C})
    scipy.io.savemat('MeanCummRegret_C.mat', mdict={'MeanCummRegret_C': MeanCummRegret_C})

if TF_plot:
    plt.figure(1)
    plt.title('Prob Opt')
    plt.plot(MeanTFArmCorrect, label='DUCB')
    plt.plot(MeanTFArmCorrect_C, label='C-DUCB')
    plt.legend()

    plt.figure(2)
    plt.title('Cummul. Regret')
    plt.plot(MeanCummRegret, label='DUCB')
    plt.plot(MeanCummRegret_C, label='C-DUCB')
    plt.legend()

    plt.show()





