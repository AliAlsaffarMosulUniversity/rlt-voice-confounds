"""Does eGeMAPS add information beyond gender? M0: gender only; M1: gender + eGeMAPS.
Conditional permutation test: shuffle eGeMAPS rows within gender (labels and gender fixed)."""
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))

import sys, numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import experiments as E, extended as X
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
EG=X.SETS['egemaps']; g=E.gender.astype(float); d=EG.shape[1]
def pipe_m1():
    ct=ColumnTransformer([("eg",Pipeline([("vt",VarianceThreshold()),("sc",StandardScaler()),("kb",SelectKBest(f_classif,k=20))]),list(range(d))),("g","passthrough",[d])])
    return Pipeline([("ct",ct),("m",LogisticRegression(max_iter=3000))]),{"m__C":[0.01,0.1,1,10]}
def pipe_m0(): return Pipeline([("m",LogisticRegression(max_iter=3000))]),{"m__C":[0.01,0.1,1,10]}
def run(model,p,s,perm=None):
    import warnings; warnings.filterwarnings("ignore")
    if model=="M0": Xm=g[:,None]; pp,gr=pipe_m0()
    else:
        eg=EG.copy()
        if perm is not None:
            rng=np.random.RandomState(500+perm)
            for k in (0,1):
                ix=np.where(g==k)[0]; eg[ix]=eg[rng.permutation(ix)]
        Xm=np.c_[eg,g]; pp,gr=pipe_m1()
    oof,pred=E.run_cv(pp,gr,Xm,p,s)
    return dict(model=model,protocol=p,seed=s,perm=-1 if perm is None else perm,acc=accuracy_score(E.y,pred),auc=roc_auc_score(E.y,oof),oof=oof.tolist(),pred=pred.tolist())
if __name__=="__main__":
    J=[(m,p,s) for m in ["M0","M1"] for p in ["random","speaker","trial"] for s in range(10)]
    J+=[("M1",p,0,i) for p in ["speaker","trial"] for i in range(200)]
    t=time.time(); out=Parallel(n_jobs=2)(delayed(run)(*j) for j in J)
    R=pd.DataFrame(out); R.drop(columns=['oof','pred']).to_csv('incremental.csv',index=False); print('%.0fs'%(time.time()-t))
