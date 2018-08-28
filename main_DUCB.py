import pandas as pd
import numpy as np
import GenData_pl as GenData
import itertools
import matplotlib.pyplot as plt
import scipy.io
import pickle
import operator


def ObserveZSim(zPossible,listPz):
    zCase = np.random.choice(len(listPz), p=listPz)
    return zPossible[zCase]

def FunEval(pl,z):
    pl_1z = pl(z)
    return [1-pl_1z, pl_1z]

def FindMaxKey(dct):
    return max(dct.items(), key=operator.itemgetter(1))[0]

def ExpectedOutcomeSim(zPossible,listPz,listEYx_z,pl,):
    sumval = 0
    for idx in range(len(listPz)):
        z = zPossible[idx]
        pz = listPz[idx]
        for x in [0, 1]:
            plval = FunEval(pl,z)
            pi_xz = plval[x]
            expected_xz = listEYx_z[idx][x]
            sumval += (pz * pi_xz * expected_xz)
    return sumval

def ComputePxz(OBS,x,zMask):
    mask = zMask & (OBS['RXASP'] == x)
    pxz = len(OBS[mask]) / len(OBS)
    return pxz
#
def ComputePz(OBS,zMask):
    pz = len(OBS[zMask]) / len(OBS)
    return pz

def ComputePzSim(listPz, zPossible, z):
    for zidx in range(len(zPossible)):
        if z[0] == zPossible[zidx][0] and z[1] == zPossible[zidx][1]:
            return listPz[zidx]


def ReceiveRewardsSim(armChosen,plChosen,listEYx_z):
    ua = listEYx_z[plChosen][armChosen]
    reward = np.random.binomial(1,ua)
    return reward

def ReceiveRewards(x,s,a,r):
    y = x + 0.5 * s + 0.2 * (r - 1) - x * a + np.random.normal(0,1)
    y = 1 / (1 + np.exp(-y))
    return y

def ComputeMpqSim(p,q,listPz):
    def f1(x):
        return x*np.exp(x-1)-1

    sumProb = 0
    for z0 in [0,1]:
        for z1 in [0,1]:
            pz = ComputePzSim(listPz, zPossible, z0, z1)
            for x in [0,1]:
                pxz = p(z0,z1)[x]
                qxz = q(z0,z1)[x]
                sumProb += (f1(pxz/qxz)*qxz*pz)

    return (1+np.log(1+sumProb))

def ComputeMpq(p,q,OBS,zName, zDomain):
    def f1(x):
        return x*np.exp(x-1)-1

    sumProb = 0
    for z in zDomain:
        zMask = (OBS[zName] == z)
        Pz = ComputePz(OBS,zMask)
        for x in [0,1]:
            pxz = FunEval(p,z)
            pxz = pxz[x]
            qxz = FunEval(q,z)
            qxz = qxz[x]
            sumProb += (f1(pxz / qxz) * qxz * Pz)

    return (1+np.log(1+sumProb))

def ComputeMatMSim(listPolicy,listPz):
    N_poly = len(listPolicy)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        pk = listPolicy[k]
        pj = listPolicy[j]
        M_mat[k,j] = ComputeMpqSim(pk,pj,listPz)
    return M_mat

def ComputeMatM(listPolicy,OBS,zName,zDomain):
    N_poly = len(listPolicy)
    poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
    M_mat = np.zeros((N_poly, N_poly))
    for k, j in poly_idx_iter:
        # if k != j:
        pk = listPolicy[k]
        pj = listPolicy[j]
        M_mat[k,j] = ComputeMpq(pk,pj,OBS,zName,zDomain)
    return M_mat

def ObserveZ(EXP):
    # s,a,r
    return list(EXP[['SEX', 'AGE', 'RCONSC']].sample(1).iloc[0])


def ComputeZkt(dictNumPolicy,nPolicy, M, k):
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        sumval += (dictNumPolicy[j] / Mkj)
    return sumval

def ComputeVal(k, j, xjs,yjs,zjs,listPolicy,M,t):
    eps = lambda t: 2 / t
    block = lambda t: 2 * np.log(t)

    Mkj = M[k, j]
    blockval = 2 * np.log(t) * Mkj

    pik = listPolicy[k]
    pij = listPolicy[j]

    pikval = FunEval(pik,zjs)
    pikval_xjs = pikval[xjs]
    pijval = FunEval(pij, zjs)
    pijval_xjs = pijval[xjs]

    invval = pikval_xjs / pijval_xjs
    if invval <= blockval:
        result = (1 / Mkj) * yjs * invval
    else:
        result = 0
    return result

def ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, M,k,t):
    eps = lambda t: 2/t
    block = lambda t: 2*np.log(t)

    nPolicy = len(listPolicy)
    Zkt = ComputeZkt(dictNumPolicy,nPolicy,M,k)
    pik = listPolicy[k]
    sumval = 0
    for j in range(nPolicy):
        Mkj = M[k,j]
        blockval = 2*np.log(t)*Mkj
        pij = listPolicy[j]
        Xj = dictdictlistPolicyData[j]['X']
        Yj = dictdictlistPolicyData[j]['Y']
        Zj = dictdictlistPolicyData[j]['Z']
        for s in range(len(Xj)):
            xjs = Xj[s]
            yjs = Yj[s]
            zjs = Zj[s]

            pikval = FunEval(pik, zjs)
            pikval_xjs = pikval[xjs]
            pijval = FunEval(pij, zjs)
            pijval_xjs = pijval[xjs]

            invval = pikval_xjs / pijval_xjs
            if invval <= blockval:
                sumval = sumval + (1/Mkj)*yjs*invval
            else:
                sumval = sumval + 0
    mu = sumval / Zkt
    return mu

def ComputeSk(dictNumPolicy, nPolicy, M, k, t):
    c1 = 16
    eps = 1e-8
    Zkt = ComputeZkt(dictNumPolicy, nPolicy, M, k)
    return np.sqrt(t*np.log(t))/(Zkt ** 2)
    # return 1.5*((np.sqrt(c1*t*np.log(t)) / Zkt)**(1/(1+eps)))

def PullArmFromPl(pl,z):
    probs = FunEval(pl,z)
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

def RunDUCB(numRound,TF_causal,TF_naive, TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB):
    ''' Variable declaration '''
    dictNumPolicy = dict() # Number of choosing policy p
    dictlistNumPolicyArm = dict()
    dictdictlistPolicyData = dict()
    dictMu = dict()
    dictSk = dict()
    dictUpper = dict()

    dictZk = dict()
    nPolicy = len(listPolicy)

    listAvgLoss = list()
    listTFArmCorrect = list()
    listCummRegret = list()

    MatSumVal = np.zeros((nPolicy, nPolicy))

    cummRegret = 0
    sumLoss = 0

    uopt = max(listU)
    optpl = np.argmax(listU)

    for plidx in range(nPolicy):
        if TF_naive  == True :
            dictMu[plidx] = np.mean(OBS['Y'])
            dictNumPolicy[plidx] = len(OBS)
        else:
            dictNumPolicy[plidx] = 0
            dictMu[plidx] = 0
        dictSk[plidx] = 0
        dictUpper[plidx] = 0
        dictZk[plidx] = 0
        dictlistNumPolicyArm[plidx] = [0, 0]
        dictdictlistPolicyData[plidx] = dict()
        dictdictlistPolicyData[plidx]['X'] = []
        dictdictlistPolicyData[plidx]['Y'] = []
        dictdictlistPolicyData[plidx]['Z'] = []

    ''' Before start'''
    # Compute the M matrix
    if TF_sim == True:
        Mmat = ComputeMatMSim(listPolicy,listPz)
    else:
        Mmat = ComputeMatM(listPolicy,OBS,'SEX',[0,1])

    ''' Initial pulling by random choice'''
    t = 1
    # Observe zj
    if TF_sim == True:
        zPossible = [[0, 0], [0, 1], [1, 0], [1, 1]]
        z = ObserveZSim(zPossible,listPz)
    else:
        z = ObserveZ(EXP)
        zObs = z[0]
    # Choose random expert j
    plidxChosen = np.random.choice(nPolicy)
    pl = listPolicy[plidxChosen] # randomly chosen policy
    # Play an arm from pl (xj)
    armChosen = PullArmFromPl(pl, zObs)
    # Receive rewards yj
    if TF_sim == True:
        reward = ReceiveRewardsSim(armChosen,plidxChosen,listEYx_z) # receive the reward
    else:
        reward = ReceiveRewards(armChosen,zObs,z[1],z[2])



    # Update information
    dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
        UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, zObs, reward)

    for k in range(nPolicy):
        # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
        MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen, yjs=reward, zjs=z, k=k, j=plidxChosen,
                                                listPolicy=listPolicy, M=Mmat, t=t)

    for k in range(nPolicy):
        dictZk[k] = ComputeZkt(dictNumPolicy,nPolicy,Mmat,k)
        dictMu[k] = ComputeMu(dictdictlistPolicyData,dictNumPolicy,listPolicy,Mmat,k,t)
        dictSk[k] = ComputeSk(dictNumPolicy,nPolicy,Mmat,k,t)
        dictUpper[k] = dictMu[k] + dictSk[k]
        if TF_causal:
            dictUpper[k] = max(min(dictUpper[k],listHB[k]),listLB[k])

    cummRegret += uopt - listU[plidxChosen]
    listCummRegret.append(cummRegret)
    if plidxChosen == optpl:
        listTFArmCorrect.append(1)
        sumLoss += 0
        avgLoss = sumLoss / t
        listAvgLoss.append(avgLoss)
    else:
        listTFArmCorrect.append(0)
        sumLoss += 1
        avgLoss = sumLoss / t
        listAvgLoss.append(avgLoss)

    # print(TF_causal, TF_naive,t)
    # print('xj,yj,zobs, z, plidx',armChosen,reward,zObs,z,plidxChosen)
    # print('Muk', dictMu)
    # print('Sk', dictSk)
    # print('Upperk', dictUpper)
    # print('Zkt', dictZk)
    # print('dicNum', dictNumPolicy)

    ''' Play! '''
    for t in range(2, numRound+1):
        # Observe Z
        if TF_sim == True:
            zPossible = [[0, 0], [0, 1], [1, 0], [1, 1]]
            z = ObserveZSim(zPossible, listPz)
        else:
            z = ObserveZ(EXP)
            zObs = z[0]

        # Choose policy
        plidxChosen = FindMaxKey(dictUpper)
        pl = listPolicy[plidxChosen]  # randomly chosen policy
        # Play an arm from pl (xj)
        armChosen = PullArmFromPl(pl, zObs)
        # Receive rewards
        if TF_sim == True:
            reward = ReceiveRewardsSim(armChosen, plidxChosen, listEYx_z)  # receive the reward
        else:
            reward = ReceiveRewards(armChosen, zObs, z[1], z[2])

        # Update information
        dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
            UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, zObs,
                           reward)
        for k in range(nPolicy):
            # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen, yjs=reward, zjs=zObs, k=k, j=plidxChosen,
                                                    listPolicy=listPolicy, M=Mmat, t=t)

        for k in range(nPolicy):
            dictZk[k] = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            if t <= 1000:
                dictMu[k] = ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, Mmat, k, t)
            else:
                dictMu[k] = 0
                for j in range(nPolicy):
                    dictMu[k] += MatSumVal[k][j]
                dictMu[k] /= dictZk[k]
            dictSk[k] = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
            dictUpper[k] = dictMu[k] + dictSk[k]
            if TF_causal:
                dictUpper[k] = max(min(dictUpper[k],listHB[k]),listLB[k])

        cummRegret += uopt - listU[plidxChosen]
        listCummRegret.append(cummRegret)
        if plidxChosen== optpl:
            listTFArmCorrect.append(1)
            sumLoss += 0
            avgLoss = sumLoss / t
            listAvgLoss.append(avgLoss)
        else:
            listTFArmCorrect.append(0)
            sumLoss += 1
            avgLoss = sumLoss / t
            listAvgLoss.append(avgLoss)

        if (t%1000 == 0):
            print(TF_causal,TF_naive,t)
            print('xj,yj,zj', armChosen, reward, z)
            print('Muk',dictMu)
            print('Sk',dictSk)
            print('Upperk',dictUpper)
            print('Zkt',dictZk)
            print('dicNum',dictNumPolicy)
    return [listTFArmCorrect, listCummRegret, listAvgLoss, dictNumPolicy,Mmat]


def RunSimulation(numSim, numRound, TF_causal, TF_naive, TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB):
    arrayTFArmCorrect = np.array([0]*numRound)
    arrayCummRegret = np.array([0]*numRound)
    arrayAvgLoss = np.array([0]*numRound)

    for k in range(numSim):
        print(k)
        listTFArmCorrect, listCummRegret, listAvgLoss, dictNumPolicy,Mmat = RunDUCB(numRound,TF_causal,TF_naive,TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB)
        # print(len(listTFArmCorrect),len(listCummRegret),len(listAvgLoss))
        arrayTFArmCorrect = arrayTFArmCorrect + np.asarray(listTFArmCorrect)
        arrayCummRegret = arrayCummRegret + np.asarray(listCummRegret)
        arrayAvgLoss = arrayAvgLoss + np.asarray(listAvgLoss)

    MeanTFArmCorrect = arrayTFArmCorrect / numSim
    MeanCummRegret = arrayCummRegret / numSim
    MeanAvgLoss = arrayAvgLoss / numSim
    return [MeanTFArmCorrect, MeanCummRegret, MeanAvgLoss,Mmat]

def Genobspl(z0,z1):
    listPxz_over_z = [[0.31481481481481477, 0.6851851851851851], [0.89738430583501, 0.10261569416498993], [0.9612590799031477, 0.038740920096852295], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    iteridx = 0
    for zidx0 in [0,1]:
        for zidx1 in [0,1,2]:
            if (z0 == zidx0) and (z1 == zidx1):
                return listPxz_over_z[iteridx]
            else:
                iteridx += 1


TF_sim = False
TF_SaveResult = False
TF_plot = True
numRound = 5000
numSim = 200

if TF_sim == True:
    ''' Simulation instance '''
    EXP = pickle.load(open('EXP.pkl', 'rb'))
    OBS = pickle.load(open('OBS.pkl', 'rb'))
    listPz = [0.1,0.2,0.3,0.4]
    zPossible = [[0,0],[0,1],[1,0],[1,1]]
    listEYx_z = [[0.7,0.2],[0.69,0.12],[0.21,0.66],[0.27,0.78]]
    listU = list()
    lowprob = 0.3
    highprob = 0.7
    pl1 = lambda z0, z1: [lowprob, highprob] if ((z0 == 0) and (z1 == 0)) else [highprob, lowprob]
    pl2 = lambda z0, z1: [lowprob, highprob] if ((z0 == 0) and (z1 == 1)) else [highprob, lowprob]
    pl3 = lambda z0, z1: [lowprob, highprob] if ((z0 == 1) and (z1 == 0)) else [highprob, lowprob]
    pl4 = lambda z0, z1: [lowprob, highprob] if ((z0 == 1) and (z1 == 1)) else [highprob, lowprob]

    listPolicy = [pl1,pl2,pl3,pl4]

    for pl in listPolicy:
        listU.append(ExpectedOutcomeSim(zPossible,listPz,listEYx_z,pl))
    optpl = np.argmax(listU)
    uopt = np.max(listU)

    listLB = [0.3,0.1,0.35,0.34]
    listHB = [0.57,0.42,0.51,0.78]
    listBdd = list()
    for pldx in range(len(listLB)):
        listBdd.append([listLB[pldx],listHB[pldx]])

    for plidx in range(len(listPolicy)):
        print('Case 2','Policy',plidx,GenData.CheckCase2(listHB[plidx],uopt))
    covZ = ['SEX', 'RCONSC']
    print(listU)
    print(listBdd)
    # print(GenData.CheckCase2())


else:
    EXP = pickle.load(open('sim_instance/EXP.pkl','rb'))
    OBS = pickle.load(open('sim_instance/OBS.pkl','rb'))
    listU = pickle.load(open('sim_instance/listU.pkl','rb'))
    listPz = pickle.load(open('sim_instance/listPz.pkl','rb'))
    listLB = pickle.load(open('sim_instance/listLB.pkl','rb'))
    listHB = pickle.load(open('sim_instance/listHB.pkl','rb'))

    # Define policies
    pl1 = lambda z: 0.01 if z == 0 else 0.02
    pl2 = lambda z: 0.05 if z == 0 else 0.1
    pl3 = lambda z: 0.97 if z == 0 else 0.99
    pl4 = lambda z: 0.1 if z == 0 else 0.05
    listPolicy = [pl1, pl2, pl3, pl4]

    uopt = np.max(listU)
    optpl = np.argmax(listU)

    listBdd = []
    for pldx in range(len(listLB)):
        listBdd.append([listLB[pldx],listHB[pldx]])

    for plidx in range(len(listPolicy)):
        print('Case 2','Policy',plidx,GenData.CheckCase2(listHB[plidx],uopt))
    print(listU)
    print(listBdd)

    # listTFArmCorrect, listCummRegret, listAvgLoss, dictNumPolicy = RunDUCB(numRound, False, False, TF_sim, covZ,
    #                                                                        EXP, listPolicy, listPz, listU, listHB,
    #                                                                        listLB)



# optpl = np.argmax(U)
# uopt = U[optpl]
# usubopt = U[1-optpl]

# [0.5915425300501339, 0.6044963160733646, 0.6506238360027363, 0.5918677614003187, 0.5981214353733352, 0.6184043428424617]
# matrix([[   1.        ,  196.77367297,  198.1685922 ,  194.7104609 ,196.88641713,  197.91989838],
#         [ 196.77367297,    1.        ,  198.36915446,  196.79650936, 197.47449037,  198.17049878],
#         [ 198.1685922 ,  198.36915446,    1.        ,  198.17430107, 198.39307074,  198.73052265],
#         [ 194.7104609 ,  196.79650936,  198.17430107,    1.        , 196.90684337,  197.92721326],
#         [ 196.88641713,  197.47449037,  198.39307074,  196.90684337, 1.        ,  198.19959516],
#         [ 197.91989838,  198.17049878,  198.73052265,  197.92721326, 198.19959516,    1.        ]])


print("-"*100)
# if TF_sim == False:
#     print("NAIVE"*100)
#     print("-"*100)
#     MeanTFArmCorrect_N, MeanCummRegret_N, MeanAvgLoss_N,Mmat_N = RunSimulation(numSim, numRound, False, True, TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB)
print("-"*100)
print("Classic"*100)
print("-"*100)
MeanTFArmCorrect, MeanCummRegret, MeanAvgLoss, Mmat= RunSimulation(numSim, numRound, False, False, TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB)
print("-"*100)
print("Causal"*100)
print("-"*100)
MeanTFArmCorrect_C, MeanCummRegret_C, MeanAvgLoss_C,Mmat_C = RunSimulation(numSim, numRound, True, False, TF_sim,EXP,listPolicy,listPz,listU,listHB,listLB)


# #
# # cut = 25000
# # cMeanTFArmCorrect = MeanTFArmCorrect[:cut]
# # cMeanTFArmCorrect_C = MeanTFArmCorrect_C[:cut]
# # cMeanCummRegret = MeanCummRegret[:cut]
# # cMeanCummRegret_C = MeanCummRegret_C[:cut]
# # cMeanAvgLoss = MeanAvgLoss[:cut]
# # cMeanAvgLoss_C = MeanAvgLoss_C[:cut]
# #
if TF_SaveResult:

    scipy.io.savemat('resultplot/MeanTFArmCorrect.mat', mdict={'MeanTFArmCorrect': MeanTFArmCorrect})
    scipy.io.savemat('resultplot/MeanCummRegret.mat', mdict={'MeanCummRegret': MeanCummRegret})
    scipy.io.savemat('resultplot/MeanAvgLoss.mat', mdict={'AvgLoss': MeanAvgLoss})
    scipy.io.savemat('resultplot/MeanTFArmCorrect_C.mat', mdict={'MeanTFArmCorrect_C': MeanTFArmCorrect_C})
    scipy.io.savemat('resultplot/MeanCummRegret_C.mat', mdict={'MeanCummRegret_C': MeanCummRegret_C})
    scipy.io.savemat('resultplot/MeanAvgLoss_C.mat', mdict={'AvgLoss_C': MeanAvgLoss_C})
    scipy.io.savemat('resultplot/MeanTFArmCorrect_N.mat', mdict={'MeanTFArmCorrect_C': MeanTFArmCorrect_N})
    scipy.io.savemat('resultplot/MeanCummRegret_N.mat', mdict={'MeanCummRegret_C': MeanCummRegret_N})
    scipy.io.savemat('resultplot/MeanAvgLoss_N.mat', mdict={'AvgLoss_C': MeanAvgLoss_N})

    pickle.dump(MeanTFArmCorrect,open('resultplot/MeanTFArmCorrect.pkl','wb'))
    pickle.dump(MeanCummRegret,open('resultplot/MeanCummRegret.pkl','wb'))
    pickle.dump(MeanAvgLoss,open('resultplot/MeanAvgLoss.pkl','wb'))
    pickle.dump(MeanTFArmCorrect_C, open('resultplot/MeanTFArmCorrect_C.pkl', 'wb'))
    pickle.dump(MeanCummRegret_C, open('resultplot/MeanCummRegret_C.pkl', 'wb'))
    pickle.dump(MeanAvgLoss_C,open('resultplot/MeanAvgLoss_C.pkl','wb'))
    pickle.dump(MeanTFArmCorrect_N, open('resultplot/MeanTFArmCorrect_N.pkl', 'wb'))
    pickle.dump(MeanCummRegret_N, open('resultplot/MeanCummRegret_N.pkl', 'wb'))
    pickle.dump(MeanAvgLoss_N,open('resultplot/MeanAvgLoss_N.pkl','wb'))

#
if TF_plot:
    # cutnum = 25000
    plt.figure(1)
    plt.title('Prob Opt')
    # if TF_sim == False:
    #     plt.plot(MeanTFArmCorrect_N, label='DUCB_N')
    plt.plot(MeanTFArmCorrect, label='DUCB')
    plt.plot(MeanTFArmCorrect_C, label='C-DUCB')
    plt.legend()

    plt.figure(2)
    plt.title('Cummul. Regret')
    # if TF_sim == False:
    #     plt.plot(MeanCummRegret_N, label='DUCB_N')
    plt.plot(MeanCummRegret, label='DUCB')
    plt.plot(MeanCummRegret_C, label='C-DUCB')
    plt.legend()

    plt.figure(3)
    plt.title('AvgLoss')
    # if TF_sim == False:
    #     plt.plot(MeanAvgLoss_N, label='DUCB_N')
    plt.plot(MeanAvgLoss, label='DUCB')
    plt.plot(MeanAvgLoss_C, label='C-DUCB')
    plt.legend()

    plt.show()
# # #
# # #
# #


