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
    terminal_cond = 1e-8
    eps = 1e-12
    if mu_hat == 1:
        return 1
    elif mu_hat == 0:
        mu_hat += eps
    mu = copy.copy(mu_hat)
    while 1:
        mu_cand = (mu + maxval) / 2
        KL_val = BinoKL(mu_hat, mu_cand)
        diff = np.abs(KL_val - M)
        # print(mu, mu_hat, mu_cand,KL_val, M, diff)
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

def ComputeDynamicMean(n,prevM,lastElem):
    return ((n-1)*prevM + lastElem)/n

def UpdateAfterArm(dictNumArm,dictM,dictLastElem,armChosen,reward):
    dictNumArm[armChosen] += 1
    dictLastElem[armChosen] = reward
    dictM[armChosen] = ComputeDynamicMean(dictNumArm[armChosen], dictM[armChosen],dictLastElem[armChosen])
    return [dictNumArm,dictM,dictLastElem]

def ComputeMu(dictlistNumArmReard, armChoice):
    return np.mean(dictlistNumArmReard[armChoice])

def RunKLUCB(listArm, listHB, listU, numRound, TF_causal, TF_sim):
    ''' Definition of variable '''
    dictNumArm = dict()
    # dictlistArmReward= dict()
    dictM = dict()
    dictLastElem = dict()
    listTFArmCorrect = list()
    listCummRegret = list()

    armOpt = np.argmax(listU)
    cummRegret = 0

    for a in listArm:
        dictNumArm[a] = 0
        dictM[a] = 0
        dictLastElem[a] = 0
        # dictlistArmReward[a] = list()

    ''' Initial pulling'''
    for a in listArm:
        if TF_sim == True:
            reward = ReceiveRewardsSim(a, listU)
        else:
            reward = ReceiveRewards(a,EXP)
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm,dictM,dictLastElem, a, reward)

    ''' Run!'''
    f = lambda x: np.log(x) + 3 * np.log(np.log(x))
    for idxround in range(numRound):
        t = idxround + len(listArm) + 1
        # Compute the mean reward
        listUpper = list()
        for a in listArm:
            # Compute
            mu_hat = dictM[a]
            ft = f(t)
            # print(t, a, mu_hat, ft, dictNumArm[a])
            upper_a = MaxKL(mu_hat,ft,dictNumArm[a],init_maxval=1)
            if TF_causal:
                upper_a = np.min([listHB[a], upper_a])
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

def RunSimulation(numSim, numRound, TF_causal,TF_sim):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunKLUCB(listArm, HB, listU, numRound, TF_causal=TF_causal, TF_sim = TF_sim)
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)
        # listlistTFArmCorrect.append(listTFArmCorrect)
        # listlistCummRegret.append(listCummRegret)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    return [MeanTFArmCorrect, MeanCummRegret]



X = 'RXASP'
IST, EXP,OBS = GenData_IST.RunGenData()
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


''' Bandit Run!'''
TF_sim = True
numRound = 2000
numSim = 100

if TF_sim == True:
    LB = [0.03, 0.21]
    HB = [0.76, 0.6]
    listU = [0.66,0.36]

listlistTFArmCorrect = list()
listlistCummRegret = list()

print("")
print("Bandit Start")
MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim,numRound,TF_causal=False, TF_sim=TF_sim)
print("-"*100)
print("")
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim,numRound,TF_causal=True,TF_sim=TF_sim)

# pickle.dump(MeanTFArmCorrect,open('MeanTFArmCorrect.pkl','wb'))
# pickle.dump(MeanCummRegret,open('MeanCummRegret.pkl','wb'))
# pickle.dump(MeanTFArmCorrect_C,open('MeanTFArmCorrect_C.pkl','wb'))
# pickle.dump(MeanCummRegret_C,open('MeanCummRegret_C.pkl','wb'))
#
scipy.io.savemat('MeanTFArmCorrect.mat', mdict={'MeanTFArmCorrect': MeanTFArmCorrect})
scipy.io.savemat('MeanCummRegret.mat', mdict={'MeanCummRegret': MeanCummRegret})
scipy.io.savemat('MeanTFArmCorrect_C.mat', mdict={'MeanTFArmCorrect_C': MeanTFArmCorrect_C})
scipy.io.savemat('MeanCummRegret_C.mat', mdict={'MeanCummRegret_C': MeanCummRegret_C})


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

# for age in [0,1]:
#     for sex in [0,1]:
#         print(len(OBS[(OBS['AGE']==age)&(OBS['SEX']==sex)])/len(OBS))

# for age in [0,1]:
#     for sex in [0,1]:
#         for x in [0,1]:
#             EYx_z = np.mean(EXP[(EXP['AGE']==age)&(EXP['SEX']==sex) & (EXP['RXASP']==x)]['Y'])
#             print('E[Yx|z]=',EYx_z, 'where X =',x,', Z =','(',age,sex,')')