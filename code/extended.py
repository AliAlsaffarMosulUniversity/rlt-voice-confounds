"""Additional feature sets: eGeMAPS (88), ComParE-2016 (6373), Resemblyzer speaker embeddings (256, if available)."""
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))

import sys, os, numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import experiments as E
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
SETS={}
for name,f in [('egemaps','egemaps.csv'),('compare','compare.csv'),('resemblyzer','resemblyzer.csv')]:
    if os.path.exists(''+f):
        X=E.M[['id']].merge(pd.read_csv(''+f),on='id').drop(columns='id').values.astype(float)
        X=np.nan_to_num(X); SETS[name]=X
def pipes():
    base=[("vt",VarianceThreshold()),("sc",StandardScaler()),("kb",SelectKBest(f_classif,k=20))]
    return {"SVM-RBF":(Pipeline(base+[("m",SVC())]),{"m__C":[0.1,1,10,100],"m__gamma":["scale",0.01,0.1]}),
            "LogReg":(Pipeline(base+[("m",LogisticRegression(max_iter=3000))]),{"m__C":[0.01,0.1,1,10]})}
def job(fs,mn,p,s):
    import warnings; warnings.filterwarnings("ignore")
    pipe,grid=pipes()[mn]; oof,pred=E.run_cv(pipe,grid,SETS[fs],p,s)
    r=E.metrics(oof,pred); r.update(fs=fs,model=mn,protocol=p,seed=s); return r
if __name__=="__main__":
    J=[(fs,mn,p,s) for fs in SETS for mn in ["SVM-RBF","LogReg"] for p in ["random","speaker","trial"] for s in range(10)]
    t=time.time(); out=Parallel(n_jobs=2)(delayed(job)(*j) for j in J)
    R=pd.DataFrame(out); R.to_csv('cv_extended.csv',index=False)
    print(R.groupby(['fs','model','protocol'])[['acc','auc']].agg(['mean','std']).round(3)); print('%.0fs'%(time.time()-t))
