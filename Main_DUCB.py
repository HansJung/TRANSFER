import pandas as pd
import numpy as np
import GenData_pl as GenData
import itertools
import matplotlib.pyplot as plt
import scipy.io


def ObserveZSim(zPossible,listPz):
    zCase = np.random.choice(len(listPz), p=listPz)
    return zPossible[zCase]

def ExpectedOutcomeSim(zPossible,listPz,listEYx_z,pl,):
    sumval = 0
    for idx in range(len(listPz)):
        z = zPossible[idx]
        pz = listPz[idx]
        for x in [0, 1]:
            pi_xz = pl(z[0], z[1])[x]
            expected_xz = listEYx_z[idx][x]
            sumval += pz * pi_xz * expected_xz
    return sumval

# def ComputePxz(OBS,age,sex,x):
#     X = 'RXASP'
#     P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
#     return P_xz
#
def ComputePz(OBS,z0,z1,covZ):
    pz = len(OBS[(OBS[covZ[0]] == z0) & (OBS[covZ[1]] == z1)]) / len(OBS)
    return pz

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


def ReceiveRewards(armChosen, zObserved, covZ, EXP):
    # sample from Y ~ E[Yx|z]
    ## Note x ~ pi(x|z)
    ## Therefore E[Yx|z] ~ E[Y|x,z] where x ~ pi(x|z) given z
    z0,z1 = zObserved[0],zObserved[1]
    y = list(EXP[(EXP[covZ[0]]==z0)&(EXP[covZ[1]]==z1)&(EXP['RXASP']==armChosen)].sample(1)['Y'])[0]
    # print(z0,z1,armChosen)
    return y

# def ReceiveRewards(plidxChosen, armChosen, listEXP):
#     X = 'RXASP'
#     EXPi = listEXP[plidxChosen]
#     reward = list(EXPi[EXPi[X] == armChosen].sample(1)['Y'])[0]
#
#     # EXPi[X == armChosen]
#     # reward = EXPi.iloc[dictlistNumPolicyArm[plidxChosen][armChosen]]['Y']
#     return reward

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

def ComputeMpq(p,q,OBS,covZ):
    def f1(x):
        return x*np.exp(x-1)-1

    sumProb = 0
    for z0 in [0,1]:
        for z1 in [0,1,2]:
            Pz = ComputePz(OBS,z0,z1,covZ)
            for x in [0,1]:
                pxz = p(z0,z1)[x]
                qxz = q(z0,z1)[x]
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

def ComputeMatM(listPolicy,OBS,covZ):
    N_poly = len(listPolicy)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        pk = listPolicy[k]
        pj = listPolicy[j]
        M_mat[k,j] = ComputeMpq(pk,pj,OBS,covZ)
    return M_mat

def ObserveZ(EXP,covZ):
    return list(EXP[covZ].sample(1).iloc[0])


def ComputeZkt(dictNumPolicy,nPolicy, M, k):
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        sumval += dictNumPolicy[j] / Mkj
    return sumval

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



# def ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, M,k,t):
#     eps = lambda t: 2/(t**2)
#     block = lambda t: 2*np.log(2/eps(t))
#
#     nPolicy = len(listPolicy)
#     Zkt = ComputeZkt(dictNumPolicy,nPolicy,M,k)
#     pik = listPolicy[k]
#     sumval = 0
#     for j in range(nPolicy):
#         Mkj = M[k,j]
#         blockval = block(t)*Mkj
#         pij = listPolicy[j]
#         Xj = dictdictlistPolicyData[j]['X']
#         Yj = dictdictlistPolicyData[j]['Y']
#         Zj = dictdictlistPolicyData[j]['Z']
#         for s in range(len(Xj)):
#             xjs = Xj[s]
#             yjs = Yj[s]
#             zjs = Zj[s]
#             invval = (pik(zjs[0], zjs[1])[xjs]/pij(zjs[0],zjs[1])[xjs])
#             if invval <= blockval:
#                 sumval += (1/Mkj)*yjs*invval
#             else:
#                 sumval += 0
#     return sumval/Zkt

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

def RunDUCB(numRound,TF_causal,TF_sim,covZ,EXP,listPolicy,listPz,listU,listHB,listLB):
    ''' Variable declaration '''
    dictNumPolicy = dict() # Number of choosing policy p

    nPolicy = len(listPolicy)
    MatSumVal = np.zeros((nPolicy,nPolicy))
    dictMu = dict()
    dictUpper = dict()
    dictSk = dict()

    listTFArmCorrect = list()

    cummRegret = 0
    listCummRegret = []

    uopt = max(listU)
    optpl = np.argmax(listU)

    for plidx in range(nPolicy):
        dictNumPolicy[plidx] = 0
        dictMu[plidx] = 0
        dictSk[plidx] = 0
        dictUpper[plidx] = 0

    ''' Before start'''
    # Compute the M matrix
    if TF_sim == True:
        Mmat = ComputeMatMSim(listPolicy)
    else:
        Mmat = ComputeMatM(listPolicy, OBS,covZ)

    ''' Initial pulling by random choice'''
    t = 1
    # Observe Z
    if TF_sim == True:
        z = ObserveZSim(listPz)
    else:
        z = ObserveZ(EXP,covZ)
    # Choose random expert
    plidxChosen = np.random.choice(nPolicy)
    pl = listPolicy[plidxChosen] # randomly chosen policy
    # Play an arm from pl
    armChosen = PullArmFromPl(pl, z)
    # Receive rewards
    if TF_sim == True:
        reward = ReceiveRewardsSim(plidxChosen,armChosen,U) # receive the reward
    else:
        reward = ReceiveRewards(armChosen,z,covZ,EXP)

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
            zPossible = [[0, 0], [0, 1], [1, 0], [1, 1]]
            z = ObserveZSim(zPossible)
        else:
            z = ObserveZ(EXP, covZ)

        # Choose policy
        if TF_causal == True:
            listUpper = list(dictUpper.values())
            listUhat = list()
            for k in range(nPolicy):
                listUhat.append(np.max([np.min([listUpper[k],listHB[k]]),listLB[k]]))
            plidxChosen = np.argmax( listUhat )
        else:
            plidxChosen = np.argmax(list(dictUpper.values()))
        pl = listPolicy[plidxChosen]
        # Play an arm
        armChosen = PullArmFromPl(pl, z)
        # Receive rewards
        if TF_sim == True:
            reward = ReceiveRewardsSim(plidxChosen, armChosen, U)
        else:
            reward = ReceiveRewards(armChosen,z,covZ,EXP)

        # Update information
        dictNumPolicy[plidxChosen] += 1
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

        cummRegret += uopt - listU[plidxChosen]
        listCummRegret.append(cummRegret)
        if plidxChosen== optpl:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        if t%1000 == 0:
            print(TF_causal,t,dictMu,dictUpper)
    return [listTFArmCorrect, listCummRegret]


def RunSimulation(numSim, numRound, TF_causal,TF_sim,covZ,EXP,listPolicy,listPz,listU,listHB,listLB):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunDUCB(numRound,TF_causal,TF_sim,covZ,EXP,listPolicy,listPz,listU,listHB,listLB)
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    return [MeanTFArmCorrect, MeanCummRegret]

TF_sim = False
TF_SaveResult = False
TF_plot = True
numRound = 5000
numSim = 100

if TF_sim == True:
    ''' Simulation instance '''
    listPz = [0.1,0.2,0.3,0.4]
    zPossible = [[0,0],[0,1],[1,0],[1,1]]
    listEYx_z = [[0.1,0.5],[0.3,0.01],[0.36,0.66],[0.27,0.72]]


    LB = [0.1,0.3]
    HB = [0.4,0.7]
    listU = list()
    listPolicy = GenData.PolicyGen()[:2]

    for pl in listPolicy:
        listU.append(ExpectedOutcomeSim(zPossible,listPz,listEYx_z,pl))
    # print(GenData.CheckCase2())

else:
    ''' Real data instance '''
    EXP, OBS, listU, listPolicy = GenData.RunGenData()
    listPz = GenData.ComputePz(EXP)
    optpl = np.argmax(listU)
    uopt = listU[optpl]

    listBdd = []
    listHB = []
    listLB = []

    for pl in listPolicy:
        LB, HB = GenData.BoundsPl(OBS, listPz, pl, X='RXASP', covZ=['SEX', 'RCONSC'])
        listBdd.append([LB, HB])
        listHB.append(HB)
        listLB.append(LB)

    for plidx in range(len(listPolicy)):
        print('Case 2','Policy',plidx,GenData.CheckCase2(listHB[plidx],uopt))

    covZ = ['SEX','RCONSC']

# optpl = np.argmax(U)
# uopt = U[optpl]
# usubopt = U[1-optpl]


MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim, numRound, False,TF_sim,covZ,EXP,listPolicy,listPz,listU,listHB,listLB)
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim, numRound, True,TF_sim,covZ,EXP,listPolicy,listPz,listU,listHB,listLB)

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
#
# #
#


