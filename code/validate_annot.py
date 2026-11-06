import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import pandas as pd, numpy as np, sys
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
M=pd.read_csv('meta.csv').merge(pd.read_csv('features.csv'),on='id')
src=sys.argv[1] if len(sys.argv)>1 else 'egemaps'
X=M[['id']].merge(pd.read_csv(src+'.csv'),on='id').drop(columns='id').values
X=StandardScaler().fit_transform(np.nan_to_num(X)) if src!='resemblyzer' else X/np.linalg.norm(X,axis=1,keepdims=True)
# automatic gender rater: F0 threshold 160 Hz
auto=np.where(M.f0_mean<160,'M','F'); k=cohen_kappa_score(M.gender,auto); agree=(auto==M.gender).mean()
print(f'gender: F0<160Hz rater agreement {agree:.3f} kappa {k:.3f}; disagreements:',list(M.id[auto!=M.gender]))
# speaker verification consistency within trials
d=[];same=[]
for t,g in M.groupby('trial'):
    ix=g.index.values
    for a in range(len(ix)):
        for b in range(a+1,len(ix)):
            i,j=ix[a],ix[b]
            dist=np.linalg.norm(X[i]-X[j]) if src!='resemblyzer' else 1-X[i]@X[j]
            d.append(-dist); same.append(M.speaker[i]==M.speaker[j])
same=np.array(same); print(f'{src}: within-trial pairs {len(same)}, same-speaker {same.sum()}, verification AUC {roc_auc_score(same,d):.3f}')
