import pandas as pd
import numpy as np
import GenData_IST_Pl as GenData

''' 
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
        - Number of choosing EXPi
        - z: observed context. 
    Proc 
        pli = listPolicy[plidx] # Probability of arm 0 and 1
        EXPi = listEXP[plidx] # Probability of arm 0 and 1
        probs = pli(z[0],z[1]) # Probability of arm 0 and 1
        armPull = np.random.binomial(1,prob[1])
        sampledEXP = EXPi[(X==armPull) & (EXPi[] == Z[0]) & (EXPi[] == Z[1])].sample(1)
        return sampledEXP['Y']
    Output:
        - Reward 

PullPolicy


'''

listEXP, OBS = GenData.RunGenData()
policy_list = GenData.PolicyGen()
GenData.QualityCheck(listEXP,OBS,policy_list)


