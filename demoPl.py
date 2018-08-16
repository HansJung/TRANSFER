def drawPolicy(pl,Z):
    prob0, prob1 = pl(Z[0],Z[1])
    return np.random.binomial(n=1, p=prob1, size=1)[0]

def SigTestPl(df,X, policy_list,alpha=0.01):
    numPolicy = len(policy_list)
    possible_pairs = list(itertools.combinations(range(numPolicy),2))

    listOutcome = []
    listStat = []

    for pl in policy_list:
        listOutcome.append(ExpectedOutcomePl(df, pl))
        listStat.append(ttestStatGen(df,X,pl))

    listPval = []
    for pair_elem in possible_pairs:
        nobs1,mean1,std1,seed1 = listStat[pair_elem[0]]
        nobs2, mean2, std2,seed2 = listStat[pair_elem[1]]
        result = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1,
                             mean2=mean2, std2=std2, nobs2=nobs2,
                             equal_var=False)

        listPval.append(result.pvalue)
    arrayPval = np.array(listPval)
    # print(possible_pairs)
    print(arrayPval)
    if sum((arrayPval < alpha) * 1) == len(listPval):
        print(seed1,seed2)
        return True
    else:
        return False

def ExpectedOutcomePl(EXP,pl):
    sum_prob = 0
    for age in [0,1]:
        for sex in [0,1]:
            Pz = len(EXP[(EXP['AGE']==age)&(EXP['SEX']==sex)])/len(EXP)
            probs = pl(age,sex)
            Yz = EXP[(EXP['AGE'] == age) & (EXP['SEX'] == sex)]
            for x in [0,1]:
                Yx_z = Yz[Yz[X]==x]['Y']
                EYx_z = np.mean(Yx_z)
                sum_prob_val = EYx_z * probs[x] * Pz
                # print(sum_prob_val, EYx_z, probs[x],Pz,age,sex)
                sum_prob += sum_prob_val
    return sum_prob

def ttestStatGen(df,X,pl):
    listSample = []
    possible_case = 2 ** 32 - 1
    seed_num = np.random.randint(possible_case)
    # weight series construction
    N = len(df)
    ## pl success condition
    # for age in [0,1]:
    #     for sex in [0,1]:
    #         prob0, prob1 = pl(age,sex)
    #         if prob1 > prob0:
    #             Z = [age,sex]
    #             break
    #
    # n = len(df[(df['AGE']==Z[0]) & (df['SEX']==Z[1])])
    # PB = 0.95
    # M = int(np.round(N/5))
    # listSampleProb = []
    # for idx in range(len(df)):
    #     elem_df = df.iloc[idx]

    np.random.seed(seed_num)
    for idx in range(len(df)):
        elem_df = df.iloc[idx]
        z = [elem_df['AGE'], elem_df['SEX']]
        x = int(elem_df[X])
        elem_probs = pl(z[0],z[1])
        sample_prob = elem_probs[x]
        if np.random.binomial(1, sample_prob, 1)[0] == 1:
            listSample.append(elem_df)
    dfSample = pd.DataFrame(listSample)
    nobs = len(listSample)
    mean_obs = np.mean(dfSample['Y'])
    std_obs = np.std(dfSample['Y'])

    return [nobs, mean_obs, std_obs, seed_num]

def SeedFindingPl(IST, X, policy_list, sample_N=12000, alpha=0.01):
    iter_idx = 0
    possible_case = 2 ** 32 - 1
    possible_pairs = list(itertools.combinations(range(len(policy_list)), 2))
    prevMaxDiff = -200
    remember_seed = 0

    # while 1:
    #     iter_idx += 1
    #     seed_num = np.random.randint(possible_case)
    #     np.random.seed(seed_num)
    #     Sample = IST.sample(n=sample_N)
    #     listOutcome = []
    #     for idx in range(len(policy_list)):
    #         pli = policy_list[idx]
    #         outcomePli = ExpectedOutcomePl(Sample, pli)
    #         listOutcome.append(outcomePli)
    #
    #     TF_sig = True
    #     listDiff = []
    #     for pair_elem in possible_pairs:
    #         diff = np.abs(listOutcome[pair_elem[0]] - listOutcome[pair_elem[1]])
    #         listDiff.append(diff)
    #         if diff < alpha:
    #             TF_sig = False
    #             break
    #     maxDiff = max(listDiff)
    #
    #     if maxDiff > prevMaxDiff:
    #         prevMaxDiff = maxDiff
    #         remember_seed = seed_num
    #     if ((iter_idx % 100) == 0):
    #         print(iter_idx, prevMaxDiff, remember_seed)
    #     if TF_sig == True:
    #         return seed_num
    #
    #     # TF_sig = SigTestPl(Sample, 'RXASP', policy_list, alpha)
    #     # print(iter_idx,TF_sig)
    #     # if TF_sig:
    #     #     return seed_num


def SeedFindingOBSPl(EXP, sample_N, policy_list, alpha=0.01): # If numPolicy = 2
    pl1, pl2 = policy_list
    outcome_pl1 = ExpectedOutcomePl(EXP, pl1)
    outcome_pl2 = ExpectedOutcomePl(EXP, pl2)
    EYpis = [outcome_pl1, outcome_pl2]

    possible_case = 2 ** 32 - 1
    iter_idx = 0
    prevMaxDiff = -200
    remember_seed = 0

    while 1:
        iter_idx += 1
        seed_num = np.random.randint(possible_case)
        np.random.seed(seed_num)
        Sample = EXP.sample(sample_N)
        LB1, HB1 = BoundsPl(Sample, pl1)
        LB2, HB2 = BoundsPl(Sample, pl2)
        HBs = [HB1, HB2]
        if CheckCase2(HBs, EYpis):
            return seed_num
        elif iter_idx % 100 == 0:
            print(iter_idx,[LB1,outcome_pl1,HB1],[LB2,outcome_pl2,HB2])


    # while 1:
    #     iter_idx += 1
    #     seed_num = np.random.randint(possible_case)
    #     np.random.seed(seed_num)
    #     Sample = EXP.sample(sample_N)
    #     listOutcome = []
    #     for idx in range(len(policy_list)):
    #         pl = policy_list[idx]
    #         outcome_pl = ExpectedOutcomePl(Sample,pl)
    #         listOutcome.append(outcome_pl)
    #
    #     diff = listOutcome[0] - listOutcome[1]
    #     if diff > prevMaxDiff:
    #         prevMaxDiff = diff
    #         remember_seed = seed_num
    #
    #     if iter_idx % 100 == 0:
    #         print(iter_idx, prevMaxDiff, remember_seed)
    #     if diff > alpha:
    #         return seed_num

def GenOBSPl(EXP,sample_N, seed_num):
    np.random.seed(seed_num)
    return EXP.sample(sample_N)

def BoundsPl(OBS,pl):
    X = 'RXASP'
    sum_prob_lb = 0
    for age in [0, 1]:
        for sex in [0, 1]:
            probs = pl(age, sex)
            for x in [0,1]:
                P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X]==x)]) / len(OBS)
                pi_xz = probs[x]
                EY_xz = np.mean(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X]==x)]['Y'])
                sum_prob_lb += EY_xz * pi_xz * P_xz

    sum_prob_ub = 0
    LB = copy.copy(sum_prob_lb)
    for age in [0, 1]:
        for sex in [0, 1]:
            probs = pl(age, sex)
            for x in [0,1]:
                P_xz = len(OBS[(OBS['AGE'] == age) & (OBS['SEX'] == sex) & (OBS[X] == x)]) / len(OBS)
                non_pi_xz = probs[1-x]
                sum_prob_ub += P_xz * non_pi_xz
    UB = LB + sum_prob_ub
    return [LB,UB]