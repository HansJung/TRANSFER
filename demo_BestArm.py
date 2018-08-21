import pandas as pd
import numpy as np
from Simulation_KLUCB import GenData_IST
import copy
import matplotlib.pyplot as plt

def ComputeMu(EXP,listArm):
    X = 'RXASP'
    listU = []
    for a in listArm:
        Ua = np.mean(EXP[EXP[X]==a]['Y'])
        listU.append(Ua)
    return listU

def FindMax2Idx(U):
    u1,u2 = FindMax2(U)
    idx_u1 = list(U).index(u1)
    idx_u2 = list(U).index(u2)
    return [idx_u1, idx_u2]


def FindMax2(U):
    Ucopy = np.array(copy.copy(U))
    Ucopy = -np.sort(-Ucopy)
    u1, u2 = Ucopy[:2]
    return [u1,u2]

def CheckCase2BestArm(HB,U):
    u1,u2 = FindMax2(U)

    hx0, hx1 = HB
    Ux0, Ux1 = U

    if Ux0 < Ux1:
        if hx0 < (u1 + u2)/2:
            return True
        else:
            return False
    else:
        if hx1 < (u1 + u2)/2:
            return True
        else:
            return False


def UpperConfidence(t,delta,eps,numArm):
    val = (1 + np.sqrt(eps)) * np.sqrt((1 + eps) * np.log(np.log((1 + eps) * t) / delta) / (2 * t))
    return val

def ReceiveReward(armChosen, EXP):
    reward = list(EXP[EXP[X] == armChosen].sample(1)['Y'])[0]
    return reward

def ComputeDynamicMean(n,prevM,lastElem):
    return ((n-1)*prevM + lastElem)/n

def UpdateAfterArm(dictNumArm,dictM,dictLastElem,armChosen,reward):
    dictNumArm[armChosen] += 1
    dictLastElem[armChosen] = reward
    dictM[armChosen] = ComputeDynamicMean(dictNumArm[armChosen], dictM[armChosen],dictLastElem[armChosen])
    return [dictNumArm,dictM,dictLastElem]

def CheckStopCondition(listMuEst,listUEst,ht,lt):
    if listMuEst[ht]-listUEst[ht] > listMuEst[lt]+listUEst[lt]:
        return True
    else:
        return False

def RunBestArm(listArm, TF_causal):
    # Note this parametrization satisfied the definition of U
    eps = 0.01
    delta = 0.01  # (1-delta) is a confidence interval
    # f = lambda eps: np.log(1+eps)/np.e

    # Declaration of variable
    numArm = len(listArm)
    listH = list()
    listProbOpt = list()
    dictNumArm = dict()
    dictM = dict()
    dictLastElem = dict()

    # dictlistArmReward = dict()
    # dictNumArmH = dict()

    for a in listArm:
        dictNumArm[a] = 1
        dictM[a] = 0
        dictLastElem[a] = 0

    # Initialization
    t = 0
    for a in listArm:
        t += 1
        reward = ReceiveReward(a, EXP)
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, a, reward)

    # Start
    print("")
    print("BestArm Start")
    while 1:
        t += 1
        listUpperEst = list()
        listMuEst = list()
        listUEst = list()
        for a in listArm:
            muEst_a = dictM[a]
            listMuEst.append(muEst_a)
            U_a = UpperConfidence(dictNumArm[a], delta, eps, numArm)
            listUEst.append(U_a)
            if TF_causal == True:
                listUpperEst.append(min(muEst_a + U_a, HB[a]))
            else:
                listUpperEst.append(muEst_a + U_a)
        # print(t,listMuEst,listUpperEst)
        ht, lt = FindMax2Idx(listUpperEst)
        listH.append(ht)
        probOpt = dictNumArm[optarm] / (t - 2)
        listProbOpt.append(probOpt)

        for a in [ht, lt]:
            reward = ReceiveReward(a, EXP)
            dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, a, reward)
        if t % 1000 == 0:
            print(t, ht, listMuEst[ht] - listUEst[ht], listMuEst[lt] + listUEst[lt])
        if CheckStopCondition(listMuEst, listUEst, ht, lt) == True:
            break

        # if t > T:
        #     break

    return t
    # print(t, ht)


X = 'RXASP'
K = 1
T = 5500
listArm = [0,1]
EXP,OBS = GenData_IST.RunGenData()
LB,HB = GenData_IST.ComputeBound(OBS,X)
U = ComputeMu(EXP,listArm)
if U[0] > U[1]:
    optarm = 0
else:
    optarm = 1

print(GenData_IST.QualityCheck(EXP,OBS,X,TF_emp=False))
print(GenData_IST.ObsEffect(EXP,'Y'))
print(GenData_IST.ObsEffect(OBS,'Y'))
print(CheckCase2BestArm(HB,U))

t= RunBestArm(listArm, TF_causal=False)
tC= RunBestArm(listArm, TF_causal=True)





