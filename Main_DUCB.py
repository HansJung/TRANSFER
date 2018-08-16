import pandas as pd
import numpy as np
import GenData_IST_Pl as GenData

listEXP, OBS = GenData.RunGenData()
policy_list = GenData.PolicyGen()
GenData.QualityCheck(listEXP,OBS,policy_list)


