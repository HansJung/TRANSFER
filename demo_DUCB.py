import pandas as pd
import numpy as np
import GenData_IST_Pl as GenData
import itertools

'''
[ALGORITHM] 
Before starting 

Initial random pulling 

Pulling  

'''

'''
[DATA]  
dictNumPolicy
- dict
- key: plidx
- val: number of policy chosen 

dictlistNumPolicyArm 
- dictlist 
- key: plidx 
- val: [number of pli chosen arm0, number of pli chosen arm1]

dictdictlistPolicyData
- dict(policy) of dict(X,Y,Z)
- data[plidx] = {X:[],Y:[],Z:[]}
- data[plidx]['X'] = []
'''


''' 
[FUNCTIONS] 
ReceiveRewards 
    Overview
        - Given a chosen policy (pi) and context (z) 
        - Draw arm from the policy (x ~ pi( |z))
        - Go to the EXPi (experimental dataset with corresponding policy)
        - Receive policy  
        
    Input
        - plidx: Chosen Policy index
        - listPolicy: List of policies  
        - listEXP: List of EXPs 
        - dictlistNumPolicyArm
        - z: observed context. 
    Proc 
        pli = listPolicy[plidx] # Probability of arm 0 and 1
        EXPi = listEXP[plidx] # Probability of arm 0 and 1
        probs = pli(z[0],z[1]) # Probability of arm 0 and 1
        armPull = np.random.binomial(1,prob[1])
        reward = EXPi.iloc(dictlistNumPolicyArm[plidx][armPull])['Y']
    Output:
        - Reward 


'''
def ComputePxz(OBS,age,sex,x):
    P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
    return P_xz

def ComputePz(OBS,age,sex):
    Pz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex)]) / len(OBS)
    return Pz

def ReceiveRewards(plidx, listPolicy, listEXP, dictlistNumPolicyArm, z):
    pli = listPolicy[plidx]  # Probability of arm 0 and 1
    EXPi = listEXP[plidx]  # Probability of arm 0 and 1
    probs = pli(z[0], z[1])  # Probability of arm 0 and 1
    armPull = np.random.binomial(1, probs[1]) # arm pull by the policy
    reward = EXPi.iloc(dictlistNumPolicyArm[plidx][armPull])['Y']
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


listEXP, OBS = GenData.RunGenData()
listPolicy = GenData.PolicyGen()
GenData.QualityCheck(listEXP,OBS,listPolicy)
Mmat = ComputeMatM(listPolicy,OBS)

''' Variable declaration '''
