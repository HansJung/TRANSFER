import numpy as np
import copy
from Simulation_KLUCB import GenData_IST
import matplotlib.pyplot as plt
import pickle
import scipy.io

def ReceiveRewardsSim(armChosen,listU):
    ua = listU[armChosen]
    reward = np.random.binomial(1,ua)
    return reward

def ReceiveRewards(armChosen, EXP):
    X = 'RXASP'
    reward = list(EXP[EXP[X] == armChosen].sample(1)['Y'])[0]
    return reward

def BinoKL(mu_hat, mu):

    if mu_hat == mu:
        return 0
    else:
        result = mu_hat * np.log(mu_hat / mu) + (1 - mu_hat) * np.log((1 - mu_hat) / (1 - mu))
    return result

def MaxBinarySearch(mu_hat, M, maxval):
    if M < 0:
        print(mu_hat,M,"ERROR")
    terminal_cond = 1e-8
    eps = 1e-12
    if mu_hat == 1:
        return 1
    elif mu_hat == 0:
        mu_hat += eps # diff
    mu = copy.copy(mu_hat)

    iteridx = 0
    while 1:
        iteridx += 1
        mu_cand = (mu + maxval) / 2
        KL_val = BinoKL(mu_hat, mu_cand)
        diff = np.abs(KL_val - M)
        # print(mu, mu_hat, mu_cand,KL_val, M, diff)
        if diff < terminal_cond:
            mu = mu_cand
            return mu

        if KL_val < M:
            mu = copy.copy(mu_cand)
        else:
            maxval = copy.copy(mu_cand)

        if np.abs(mu-maxval) < terminal_cond:
            return mu

        # if iteridx > 2000:
        #     print(mu_hat, M, mu, "ERROR")
        #     return mu

        # if np.abs(mu - 1) < terminal_cond:
        #     return mu

def MaxKL(mu_hat, ft, NaT, init_maxval=1):
    maxval = copy.copy(init_maxval)
    M = ft/NaT
    mu = MaxBinarySearch(mu_hat, M, maxval)
    return mu

def ComputeDynamicMean(n,prevM,lastElem):
    return ((n-1)*prevM + lastElem)/n

def UpdateAfterArm(dictNumArm,dictM,dictLastElem,armChosen,reward):
    dictNumArm[armChosen] += 1
    dictLastElem[armChosen] = reward
    dictM[armChosen] = ComputeDynamicMean(dictNumArm[armChosen], dictM[armChosen],dictLastElem[armChosen])
    return [dictNumArm,dictM,dictLastElem]

def ComputeMu(dictlistNumArmReard, armChoice):
    return np.mean(dictlistNumArmReard[armChoice])

def RunKLUCB(listArm, listHB, listLB, listU, numRound, TF_causal, TF_naive, TF_sim):
    ''' Definition of variable '''
    dictNumArm = dict() # Number of pulling arm a
    dictM = dict() # Average of reward of arm a
    dictLastElem = dict() # previous reward of arm a
    listTFArmCorrect = list() # 1 if arm a = optimal arm // 0 otherwise.
    listCummRegret = list() # cummulative regret += E[Y|do(X=optimal)] - E[Y|do(X=a)]

    armOpt = np.argmax(listU)
    cummRegret = 0

    for a in listArm:
        if TF_naive == True:
            dictNumArm[a] = listnOBS[a]
            dictM[a] = listOBSU[a]
            dictLastElem[a] = 0
        else:
            dictNumArm[a] = 0
            dictM[a] = 0
            dictLastElem[a] = 0

    ''' Initial pulling'''
    # Pulling all arm at once.
    for a in listArm:
        if TF_sim == True:
            reward = ReceiveRewardsSim(a, listU)
        else:
            reward = ReceiveRewards(a,EXP)
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm,dictM,dictLastElem, a, reward)
        cummRegret += listU[armOpt] - listU[a]
        listCummRegret.append(cummRegret)
        if a == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)

    ''' Run!'''
    f = lambda x: np.log(x) + 3 * np.log(np.log(x))
    for idxround in range(numRound-2):
        t = idxround + len(listArm) + 1 # t=3,4,...,nRound+2 // total nRound.
        # Compute the mean reward
        listUpper = list() # Each arm's upper confidence.
        for a in listArm:
            # Compute
            mu_hat = dictM[a] # Average rewards of arm a up to (t-1)
            ft = f(t)
            # print(t, a, mu_hat, ft, dictNumArm[a])
            upper_a = MaxKL(mu_hat,ft,dictNumArm[a],init_maxval=1) # argmax_u KL(mu_hat, u) < (ft/Na(t)) s.t. 0<= u <= 1.
            if TF_causal:
                upper_a = np.max([np.min([listHB[a], upper_a]),listLB[a]])
            listUpper.append(upper_a)
        # print(t,listUpper)
        armChosen = np.argmax(listUpper)
        if TF_sim == True:
            reward = ReceiveRewardsSim(armChosen, listU)
        else:
            reward = ReceiveRewards(armChosen, EXP)
        cummRegret += listU[armOpt] - listU[armChosen]
        if armChosen == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        listCummRegret.append(cummRegret)
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, armChosen, reward)

    return listTFArmCorrect, listCummRegret

def RunSimulation(numSim, numRound,HB,LB, TF_causal,TF_naive, TF_sim):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunKLUCB(listArm, listHB=HB, listLB=LB, listU=listU, numRound=numRound, TF_causal=TF_causal, TF_naive=TF_naive, TF_sim = TF_sim)
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    return [MeanTFArmCorrect, MeanCummRegret]

TF_sim = False
TF_emp = False
TF_SaveResult = False
TF_plot = True

numRound = 2000
numSim = 100

if TF_sim == True:
    ''' Externally provided simulation instances '''
    LB = [0.03, 0.21]
    HB = [0.76, 0.6]
    listOBSU = [0.4, 0.5]
    listU = [0.66, 0.36]
    listArm = [0, 1]
    # nOBS = 500
    listnOBS = [100,100]
    print(GenData_IST.CheckCase2(HB,listU))

else:
    ''' Real data'''
    EXP, OBS = GenData_IST.RunGenData()
    # OBS = OBS.sample(n=int(len(OBS)/10))
    listnOBS = [len(OBS[OBS['RXASP']==0]),len(OBS[OBS['RXASP']==1])]
    listOBSU = GenData_IST.ComputeEffect(OBS, 'RXASP', 'Y')
    # listnOBS = [100,500]
    # listnOBS = [50000,50000]
    # listnOBS = [5905000,79000]
    # print(GenData_IST.QualityCheck(EXP,OBS,'RXASP',TF_emp=False))
    # print('EXP', GenData_IST.ComputeEffect(EXP,'RXASP','Y'))
    # print('OBS',GenData_IST.ComputeEffect(OBS,'RXASP','Y'))

    LB,HB = GenData_IST.ComputeBound(OBS,'RXASP')
    listArm = [0,1]
    listU = GenData_IST.ComputeEffect(EXP,'RXASP','Y')
    nOBS = len(OBS)
    LBe, HBe = GenData_IST.EmpiricalComputeBound(OBS, 'RXASP', delta=0.01, N=nOBS)
    # listOBSU = [listU[1],listU[0]]
    print('Case 2 (true)?', GenData_IST.CheckCase2(HB, listU))
    print('Case 2 (empirical)?',GenData_IST.CheckCase2(HBe, listU))
    print(LB,HB)
    print(LBe,HBe)

print("")
print("Bandit: Naive")
MeanTFArmCorrect_N, MeanCummRegret_N = RunSimulation(numSim,numRound,HB=HB,LB=LB,TF_causal=False, TF_naive = True, TF_sim=TF_sim)
print("-"*100)
# print("")
# print("Bandit: KLUCB-")
# print(LBe,HBe)
# MeanTFArmCorrect_e, MeanCummRegret_e = RunSimulation(numSim,numRound, HB=HBe, LB=LBe, TF_causal=True, TF_naive = False, TF_sim=TF_sim)
print("Bandit: KLUCB")
MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim,numRound, HB=HB, LB=LB, TF_causal=False, TF_naive = False, TF_sim=TF_sim)
print("-"*100)
print("Bandit: C-KLUCB")
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim,numRound,HB=HB, LB=LB,TF_causal=True, TF_naive = False, TF_sim=TF_sim)

if TF_SaveResult:
    # pickle.dump(MeanTFArmCorrect,open('MeanTFArmCorrect.pkl','wb'))
    # pickle.dump(MeanCummRegret,open('MeanCummRegret.pkl','wb'))
    # pickle.dump(MeanTFArmCorrect_C,open('MeanTFArmCorrect_C.pkl','wb'))
    # pickle.dump(MeanCummRegret_C,open('MeanCummRegret_C.pkl','wb'))

    scipy.io.savemat('MeanTFArmCorrect.mat', mdict={'MeanTFArmCorrect': MeanTFArmCorrect})
    scipy.io.savemat('MeanCummRegret.mat', mdict={'MeanCummRegret': MeanCummRegret})
    scipy.io.savemat('MeanTFArmCorrect_C.mat', mdict={'MeanTFArmCorrect_C': MeanTFArmCorrect_C})
    scipy.io.savemat('MeanCummRegret_C.mat', mdict={'MeanCummRegret_C': MeanCummRegret_C})

if TF_plot == True:
    plt.figure(1)
    plt.title('Prob Opt')
    plt.plot(MeanTFArmCorrect_N,label='KLUCB-Naive')
    plt.plot(MeanTFArmCorrect,label='KLUCB')
    plt.plot(MeanTFArmCorrect_C, label='B-KLUCB')
    plt.legend()

    plt.figure(2)
    plt.title('Cummul. Regret')
    plt.plot(MeanCummRegret_N, label='KLUCB-Naive')
    plt.plot(MeanCummRegret,label='KLUCB')
    plt.plot(MeanCummRegret_C, label='B-KLUCB')
    plt.legend()

    plt.show()
