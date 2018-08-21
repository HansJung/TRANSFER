import numpy as np
import copy
from scipy.optimize import minimize
from UCB import UCB
from Simulation_KLUCB.KLUCB import KLUCB
from Simulation_KLUCB import GenData_IST
import matplotlib.pyplot as plt

def ReceiveRewards(armChosen, EXP):
    X = 'RXASP'
    reward = list(EXP[EXP[X] == armChosen].sample(1)['Y'])[0]
    return reward

def BinoKL(mu_hat, mu):
    return mu_hat * np.log(mu_hat / mu) + (1 - mu_hat) * np.log((1 - mu_hat) / (1 - mu))

def MaxBinarySearch(mu_hat, M, maxval):
    terminal_cond = 1e-8
    mu = copy.copy(mu_hat)
    while 1:
        mu_cand = (mu + maxval) / 2
        KL_val = BinoKL(mu_hat, mu_cand)
        diff = np.abs(KL_val - M)
        if KL_val < M:
            if diff < terminal_cond:
                mu = mu_cand
                return mu
            else:
                mu = copy.copy(mu_cand)
        else:
            maxval = copy.copy(mu_cand)
        if np.abs(mu - 1) < terminal_cond:
            return mu

def MaxKL(mu_hat, ft, NaT, init_maxval=1):
    maxval = copy.copy(init_maxval)
    M = ft/NaT
    mu = MaxBinarySearch(mu_hat, M,maxval)
    return mu

def UpdateAfterArm(dictNumArm, dictlistArmReward,armChosen,reward):
    dictNumArm[armChosen] += 1
    dictlistArmReward[armChosen].append(reward)
    return [dictNumArm, dictlistArmReward]

def ComputeMu(dictlistNumArmReard, armChoice):
    return np.mean(dictlistNumArmReard[armChoice])

def RunKLUCB(listArm, listHB, listU, numRound, TF_causal):
    ''' Definition of variable '''
    dictNumArm = dict()
    dictlistArmReward= dict()
    listTFArmCorrect = list()
    listCummRegret = list()
    armOpt = np.argmax(listU)
    cummRegret = 0

    for a in listArm:
        dictNumArm[a] = 0
        dictlistArmReward[a] = list()

    ''' Initial pulling'''
    for a in listArm:
        reward = ReceiveRewards(a,EXP)
        dictNumArm, dictlistArmReward = UpdateAfterArm(dictNumArm,dictlistArmReward,a,reward)

    ''' Run!'''
    f = lambda x: np.log(x) + 3 * np.log(np.log(x))
    for idxround in range(numRound):
        t = idxround + len(listArm) + 1
        # Compute the mean reward
        listUpper = list()
        for a in listArm:
            mu_hat = ComputeMu(dictlistArmReward,a)
            ft = f(t)
            upper_a = MaxKL(mu_hat,ft,dictNumArm[a],init_maxval=1)
            if TF_causal:
                upper_a = np.min([listHB[a], upper_a])
            listUpper.append(upper_a)
        armChosen = np.argmax(listUpper)
        reward = ReceiveRewards(armChosen, EXP)
        cummRegret += listU[armOpt] - listU[armChosen]
        if armChosen == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        listCummRegret.append(cummRegret)
        dictNumArm, dictlistArmReward = UpdateAfterArm(dictNumArm, dictlistArmReward, armChosen, reward)

    return listTFArmCorrect, listCummRegret

def RunSimulation(numSim, numRound, TF_causal):
    listlistTFArmCorrect = list()
    listlistCummRegret = list()

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunKLUCB(listArm, HB, listU, numRound, TF_causal=TF_causal)
        listlistTFArmCorrect.append(listTFArmCorrect)
        listlistCummRegret.append(listCummRegret)

    MatTFArmCorrect = np.matrix(listlistTFArmCorrect)
    MatCummRegret = np.matrix(listlistCummRegret)
    MeanTFArmCorrect = np.array(np.mean(MatTFArmCorrect, axis=0))[0]
    MeanCummRegret = np.array(np.mean(MatCummRegret, axis=0))[0]
    return [MeanTFArmCorrect, MeanCummRegret]



X = 'RXASP'
EXP,OBS = GenData_IST.RunGenData()
print(GenData_IST.QualityCheck(EXP,OBS,X,TF_emp=True))
print("")
print(GenData_IST.QualityCheck(EXP,OBS,X,TF_emp=False))
print(GenData_IST.ObsEffect(EXP,'Y'))
print(GenData_IST.ObsEffect(OBS,'Y'))

# LB,HB = GenData_IST.EmpiricalComputeBound(OBS,X,delta=0.01)
LB,HB = GenData_IST.ComputeBound(OBS,X)
lx0,lx1 = LB
hx0,hx1 = HB
bound_list = [[lx0,hx0],[lx1,hx1]]
listArm = [0,1]
listU = GenData_IST.ObsEffect(EXP,'Y')

# EXP = GenData_IST.ChangeRXASPtoX(EXP,idx_X=2)
# OBS = GenData_IST.ChangeRXASPtoX(OBS,idx_X=2)

''' Bandit Run!'''
numRound = 5000
numSim = 100

listlistTFArmCorrect = list()
listlistCummRegret = list()

MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim,numRound,TF_causal=False)
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim,numRound,TF_causal=True)

plt.figure(1)
plt.title('Prob Opt')
plt.plot(MeanTFArmCorrect,label='KLUCB')
plt.plot(MeanTFArmCorrect_C, label='B-KLUCB')
plt.legend()

plt.figure(2)
plt.title('Cummul. Regret')
plt.plot(MeanCummRegret,label='KLUCB')
plt.plot(MeanCummRegret_C, label='B-KLUCB')
plt.legend()

plt.show()
