import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
M=pd.read_csv('meta.csv').merge(pd.read_csv('features.csv'),on='id'); y=(M.label=='lie').astype(int).values; spk=M.speaker.values
O=json.load(open('oof.json')); R=pd.read_csv('cv_results.csv')
def cboot(oof,pred,B=2000,seed=0):
    rng=np.random.RandomState(seed); us=np.unique(spk); idx={s:np.where(spk==s)[0] for s in us}; A=[];U=[]
    for _ in range(B):
        ii=np.concatenate([idx[s] for s in rng.choice(us,len(us))])
        if len(set(y[ii]))<2: continue
        A.append(accuracy_score(y[ii],pred[ii])); U.append(roc_auc_score(y[ii],oof[ii]))
    return np.percentile(A,[2.5,97.5]),np.percentile(U,[2.5,97.5])
rows=[]
for fs,mn,p,s,oof,pred in O:
    if s!=0: continue
    a,u=cboot(np.array(oof),np.array(pred))
    rows.append(dict(fs=fs,model=mn,protocol=p,acc_lo=a[0],acc_hi=a[1],auc_lo=u[0],auc_hi=u[1]))
C=pd.DataFrame(rows)
T=R.groupby(['fs','model','protocol'])[['acc','auc','f1']].agg(['mean','std']); T.columns=['_'.join(c) for c in T.columns]; T=T.reset_index().merge(C,on=['fs','model','protocol'])
T.round(3).to_csv('table_main.csv',index=False)
pd.set_option('display.width',250); print(T[(T.fs=='speech')|(T.model=='LogReg')|(T.fs=='speech_gendernorm')].round(3).to_string())
# within-speaker F0 descriptive for mixed-label speakers
for s in ['S29', 'S02', 'S03', 'S16', 'S45']:
    d=M[M.speaker==s]; print(s, d.groupby('label').f0_mean.agg(['mean','count']).round(1).to_dict())
# figure 1: accuracy by protocol for each model (speech) + baselines
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ms=['SVM-RBF','LogReg','RandomForest','GradBoost','MLP']; P=['random','speaker','trial']; cols={'random':'#D55E00','speaker':'#0072B2','trial':'#009E73'}
x=np.arange(len(ms))
for i,p in enumerate(P):
    v=[R[(R.fs=='speech')&(R.model==m)&(R.protocol==p)].acc for m in ms]
    ax[0].bar(x+(i-1)*0.26,[a.mean() for a in v],0.26,yerr=[a.std() for a in v],color=cols[p],label={'random':'Random 10-fold','speaker':'Speaker-grouped','trial':'Trial-grouped'}[p],capsize=2)
ax[0].axhline(61/121,ls=':',c='k',lw=1); ax[0].text(4.45,61/121+.005,'majority',fontsize=7,ha='right')
ax[0].axhline(0.686,ls='--',c='#555',lw=1); ax[0].text(4.45,0.69,'gender rule',fontsize=7,ha='right')
ax[0].axhline(0.777,ls='-.',c='#999',lw=1); ax[0].text(4.45,0.78,'role rule',fontsize=7,ha='right')
ax[0].set_xticks(x); ax[0].set_xticklabels(ms,fontsize=8); ax[0].set_ylim(0.35,0.85); ax[0].set_ylabel('Accuracy (mean ± SD, 10 repeats)'); ax[0].set_title('(a) 47 speech features, nested CV',fontsize=9); ax[0].legend(fontsize=7,frameon=True,loc='lower left',bbox_to_anchor=(0,0.0))
lab={'speech':'Speech\nfeatures','speech_gendernorm':'Speech, gender-\nnormalised','f0_only':'Mean F0\nonly','channel':'Channel\n(non-speech) only'}
k=list(lab); x=np.arange(len(k))
for i,p in enumerate(P):
    v=[R[(R.fs==f)&(R.model=='LogReg')&(R.protocol==p)].acc for f in k]
    ax[1].bar(x+(i-1)*0.26,[a.mean() for a in v],0.26,yerr=[a.std() for a in v],color=cols[p],capsize=2)
ax[1].axhline(61/121,ls=':',c='k',lw=1); ax[1].axhline(0.686,ls='--',c='#555',lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([lab[f] for f in k],fontsize=7.5); ax[1].set_ylim(0.35,0.85); ax[1].set_title('(b) Logistic regression by feature set',fontsize=9)
plt.tight_layout(); plt.savefig('fig1.png',dpi=220)
# figure 2: F0 by gender and label
fig,ax=plt.subplots(figsize=(4.4,3.2))
for j,(g,l) in enumerate([('F','lie'),('F','truth'),('M','lie'),('M','truth')]):
    v=M[(M.gender==g)&(M.label==l)].f0_mean; ax.scatter(np.full(len(v),j)+np.random.RandomState(j).uniform(-.15,.15,len(v)),v,s=10,alpha=.7,c='#D55E00' if l=='lie' else '#0072B2')
    ax.plot([j-.25,j+.25],[v.median()]*2,c='k')
ax.set_xticks(range(4)); ax.set_xticklabels([f'Female\nlie (n=49)','Female\ntruth (n=26)','Male\nlie (n=12)','Male\ntruth (n=34)'],fontsize=8); ax.set_ylabel('Mean F0 (Hz)')
plt.tight_layout(); plt.savefig('fig2.png',dpi=220)
