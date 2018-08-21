# Generated with SMOP  0.41
from libsmop import *
# best_arm.m

clear('all')
close_('all')
#h = [0.55, 0.75, 0.56, 0.59];
#l = [0.3, 0.6, 0.5, 0.4];
#miu = [0.5, 0.65, 0.55, 0.5];

#h = [0.85, 0.86];
#l = [0.35, 0.4];
#miu = [0.7, 0.45];

#h = [0.8, 0.8, 0.8];
#l = [0.4, 0.4, 0.4];
#miu = [0.4, 0.5, 0.59];

#h = [0.6795, 0.5455];
#l = [0.0395, 0.1844];
#miu = [0.6115, 0.3573];

#h = [0.49, 0.50];
#l = [0.48, 0.48];
#miu = [0.48, 0.50];

h=concat([0.34,0.7])
# best_arm.m:24
l=concat([0.3,0.15])
# best_arm.m:25
miu=concat([0.1,0.6])
# best_arm.m:26
#h = [0.4, 0.8];
#l = [0.0, 0.48];
#miu = [0.3, 0.49];

#h = [0.75, 1];
#l = [0.3, 0.4];
#miu = [0.5, 0.7];

#h = [0.55, 0.8];
#l = [0.3, 0.4];
#miu = [0.5, 0.6];

#h = [0.7, 0.8];
#l = [0.4, 0.6];
#miu = [0.5, 0.65];

#h = [0.8, 0.7];
#l = [0.4, 0.3];
#miu = [0.5, 0.65];

#h = [0.6, 1];
#l = [0.4, 0.3];
#miu = [0.5, 0.7];

#h = [0.8, 0.85];
#l = [0.4, 0.3];
#miu = [0.5, 0.65];

T=100
# best_arm.m:56
K=10
# best_arm.m:57
N=length(h)
# best_arm.m:58
e=0.01
# best_arm.m:60
d=0.01
# best_arm.m:61
rounds=0
# best_arm.m:63
acc=0
# best_arm.m:64
for k in arange(1,K).reshape(-1):
    #k
    counts=zeros(N,2)
# best_arm.m:68
    round=0
# best_arm.m:69
    ht=1
# best_arm.m:70
    while 1:

        if round < N:
            x=round + 1
# best_arm.m:74
            y=binornd(1,miu(x))
# best_arm.m:76
            counts[x,1]=counts(x,1) + y
# best_arm.m:78
            counts[x,2]=counts(x,2) + 1 - y
# best_arm.m:79
        else:
            theta=zeros(1,N)
# best_arm.m:81
            value=zeros(1,N)
# best_arm.m:82
            for i in arange(1,N).reshape(-1):
                theta[i]=counts(i,1) / (counts(i,1) + counts(i,2))
# best_arm.m:85
                value[i]=theta(i) + fun5(counts(i,1) + counts(i,2),d / N,e)
# best_arm.m:86
            val,ht=max(theta,nargout=2)
# best_arm.m:89
            value[ht]=- Inf
# best_arm.m:90
            val,lt=max(value,nargout=2)
# best_arm.m:91
            if theta(ht) - fun5(counts(ht,1) + counts(ht,2),d / N,e) > val:
                break
            y=binornd(1,miu(ht))
# best_arm.m:97
            counts[ht,1]=counts(ht,1) + y
# best_arm.m:99
            counts[ht,2]=counts(ht,2) + 1 - y
# best_arm.m:100
            y=binornd(1,miu(lt))
# best_arm.m:102
            counts[lt,1]=counts(lt,1) + y
# best_arm.m:104
            counts[lt,2]=counts(lt,2) + 1 - y
# best_arm.m:105
        round=round + 1
# best_arm.m:108

    rounds=rounds + round
# best_arm.m:111
    val,opt=max(miu,nargout=2)
# best_arm.m:112
    acc=acc + (ht == opt)
# best_arm.m:113

rounds2=0
# best_arm.m:116
acc2=0
# best_arm.m:117
for k in arange(1,K).reshape(-1):
    #k
    counts=zeros(N,2)
# best_arm.m:120
    round=0
# best_arm.m:121
    ht=1
# best_arm.m:122
    while 1:

        if round < N:
            x=round + 1
# best_arm.m:126
            y=binornd(1,miu(x))
# best_arm.m:128
            counts[x,1]=counts(x,1) + y
# best_arm.m:130
            counts[x,2]=counts(x,2) + 1 - y
# best_arm.m:131
        else:
            theta=zeros(1,N)
# best_arm.m:133
            value=zeros(1,N)
# best_arm.m:134
            for i in arange(1,N).reshape(-1):
                theta[i]=counts(i,1) / (counts(i,1) + counts(i,2))
# best_arm.m:137
                value[i]=max(min(theta(i) + fun5(counts(i,1) + counts(i,2),d / N,e),h(i)),l(i))
# best_arm.m:138
            val,ht=max(value,nargout=2)
# best_arm.m:141
            value[ht]=- Inf
# best_arm.m:142
            val,lt=max(value,nargout=2)
# best_arm.m:143
            if max(min(theta(ht) - fun5(counts(ht,1) + counts(ht,2),d / N,e),h(ht)),l(ht)) > val:
                break
            y=binornd(1,miu(ht))
# best_arm.m:149
            counts[ht,1]=counts(ht,1) + y
# best_arm.m:151
            counts[ht,2]=counts(ht,2) + 1 - y
# best_arm.m:152
            y=binornd(1,miu(lt))
# best_arm.m:154
            counts[lt,1]=counts(lt,1) + y
# best_arm.m:156
            counts[lt,2]=counts(lt,2) + 1 - y
# best_arm.m:157
        round=round + 1
# best_arm.m:160

    rounds2=rounds2 + round
# best_arm.m:163
    val,opt=max(miu,nargout=2)
# best_arm.m:164
    acc2=acc2 + (ht == opt)
# best_arm.m:165

rounds=rounds / K
# best_arm.m:168
acc=acc / K
# best_arm.m:169
rounds2=rounds2 / K
# best_arm.m:171
acc2=acc2 / K
# best_arm.m:172