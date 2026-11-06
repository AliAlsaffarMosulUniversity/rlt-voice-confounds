import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LogisticRegression
M=pd.read_csv('meta.csv').merge(pd.read_csv('features.csv'),on='id'); y=(M.label=='lie').astype(int)
wit=M.role.isin(['witness','victim','prosecutor witness'])
print('role rule acc', ((~wit).astype(int)==y).mean().round(3))
print('gender rule acc', ((M.gender=='F').astype(int)==y).mean().round(3))
SP=[c for c in M.columns[M.columns.get_loc('f0_mean'):] if not c.startswith('ch_')]
CH=[c for c in M.columns if c.startswith('ch_')]
cnt=M.speaker.value_counts(); keep=M.speaker.isin(cnt[cnt>=2].index)
X=StandardScaler().fit_transform(M.loc[keep,SP]); s=np.asarray(M.loc[keep,'speaker'],dtype=object)
p=cross_val_predict(KNeighborsClassifier(1),X,s,cv=LeaveOneOut())
print('speaker-ID 1NN acc',(p==s).mean().round(3),'n',keep.sum(),'speakers',len(set(s)),'majority',round(cnt[cnt>=2].max()/keep.sum(),3))
cntt=M.trial.value_counts(); kt=M.trial.isin(cntt[cntt>=2].index)
Xc=StandardScaler().fit_transform(M.loc[kt,CH]); t=np.asarray(M.loc[kt,'trial'],dtype=object)
pc=cross_val_predict(KNeighborsClassifier(1),Xc,t,cv=LeaveOneOut()); print('trial-ID from channel 1NN',(pc==t).mean().round(3),'n',kt.sum(),'trials',len(set(t)),'majority',round(cntt.max()/kt.sum(),3))
pg=cross_val_predict(LogisticRegression(max_iter=2000),StandardScaler().fit_transform(M[SP]),np.asarray(M.gender,dtype=object),cv=10); print('gender from speech feats',(pg==M.gender).mean().round(3))
