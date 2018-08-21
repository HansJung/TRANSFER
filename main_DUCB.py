import pandas as pd
import numpy as np
import GenData_IST_Pl as GenData
import itertools
import matplotlib.pyplot as plt
import scipy.io

def ComputePxz(OBS,age,sex,x):
    X = 'RXASP'
    P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
    return P_xz

def ComputePz(OBS,age,sex):
    Pz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex)]) / len(OBS)
    return Pz

def ReceiveRewards(plidxChosen, armChosen, listEXP):
    X = 'RXASP'
    EXPi = listEXP[plidxChosen]
    reward = list(EXPi[EXPi[X] == armChosen].sample(1)['Y'])[0]

    # EXPi[X == armChosen]
    # reward = EXPi.iloc[dictlistNumPolicyArm[plidxChosen][armChosen]]['Y']
    return reward

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
    return list(IST.iloc[t-1][['AGE', 'SEX']])

def ComputeZkt(dictNumPolicy,nPolicy, M, k):
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        sumval += dictNumPolicy[j] / Mkj
    return sumval

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
    Zkt = ComputeZkt(dictNumPolicy, nPolicy, M, k)
    return 1.5*np.sqrt(c1*t*np.log(t)) / Zkt

def PullArmFromPl(pl,z):
    probs = pl(z[0],z[1])
    return np.random.binomial(1,probs[1])

def UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm,dictdictlistPolicyData,plidxChosen,armChosen,z,rewardReceived):
    dictNumPolicy[plidxChosen] += 1
    dictlistNumPolicyArm[plidxChosen][armChosen] += 1
    dictdictlistPolicyData[plidxChosen]['Z'].append(z)
    dictdictlistPolicyData[plidxChosen]['X'].append(armChosen)
    dictdictlistPolicyData[plidxChosen]['Y'].append(rewardReceived)
    return [dictNumPolicy,dictlistNumPolicyArm,dictdictlistPolicyData]

def UpdateOptProbAfterArm(dictNumPolicy,optpl,t):
    probOptPlChoose = dictNumPolicy[optpl]/t
    return probOptPlChoose

def RunDUCB(numRound,TF_causal):
    ''' Variable declaration '''
    dictNumPolicy = dict()
    dictlistNumPolicyArm = dict()
    dictdictlistPolicyData = dict()

    listProbOpt = []

    cummRegret = 0
    listCummRegret = []

    nPolicy = len(listPolicy)
    for plidx in range(nPolicy):
        dictNumPolicy[plidx] = 0
        dictlistNumPolicyArm[plidx] = [0, 0]
        dictdictlistPolicyData[plidx] = dict()
        dictdictlistPolicyData[plidx]['X'] = []
        dictdictlistPolicyData[plidx]['Y'] = []
        dictdictlistPolicyData[plidx]['Z'] = []

    ''' Before start'''
    Mmat = ComputeMatM(listPolicy, OBS)

    ''' Initial pulling by choosing random '''
    t = 1
    # Observe Z
    z = ObserveZ(IST, t)
    # Choose random expert
    plidxChosen = np.random.binomial(1, 0.5)
    pl = listPolicy[plidxChosen]
    # Play an arm
    armChosen = PullArmFromPl(pl, z)
    # Receive rewards
    reward = ReceiveRewards(plidxChosen, armChosen, listEXP)
    # Update information
    dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
        UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, z, reward)
    probOptPlChoose = UpdateOptProbAfterArm(dictNumPolicy, optpl, t)

    ''' Play! '''
    for t in range(2, numRound):
        # Observe Z
        z = ObserveZ(IST, t)
        listU = []
        for k in range(nPolicy):
            muk = ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, Mmat, k, t)
            sk = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
            Uk = muk + sk
            if TF_causal:
                Uk = np.min([Uk, HB[k]])
            listU.append(Uk)
        # Choose Policy
        plidxChosen = np.argmax(listU)
        pl = listPolicy[plidxChosen]
        # Play an arm
        armChosen = PullArmFromPl(pl, z)
        # Receive rewards
        reward = ReceiveRewards(plidxChosen, armChosen, listEXP)
        # Update information
        dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
            UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, z,
                           reward)
        probOptPlChoose = UpdateOptProbAfterArm(dictNumPolicy, optpl, t)
        listProbOpt.append(probOptPlChoose)
        cummRegret += uopt - U[plidxChosen]
        listCummRegret.append(cummRegret)
    return [listProbOpt, listCummRegret]

def RunSimulation(numSim, numRound, TF_causal):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret = RunDUCB(numRound,TF_causal)
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    return [MeanTFArmCorrect, MeanCummRegret]

# if __name__ == '__main__':
listEXP, OBS, IST = GenData.RunGenData()
listPolicy = GenData.PolicyGen()
LB,U,HB = GenData.QualityCheck(listEXP,OBS,listPolicy)
optpl = np.argmax(U)
uopt = U[optpl]
usubopt = U[1-optpl]

numRound = 500
numSim = 100
MeanTFArmCorrect, MeanCummRegret = RunSimulation(numSim, numRound, TF_causal=False)
MeanTFArmCorrect_C, MeanCummRegret_C = RunSimulation(numSim, numRound, TF_causal=True)

# scipy.io.savemat('listProbOpt.mat', mdict={'listProbOpt': listProbOpt})
# scipy.io.savemat('listProbOpt_C.mat', mdict={'listProbOpt_C': listProbOpt_C})
# scipy.io.savemat('listCummRegret.mat', mdict={'listCummRegret': listCummRegret})
# scipy.io.savemat('listCummRegret_C.mat', mdict={'listCummRegret_C': listCummRegret_C})

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
