"""Gender-normalised versions of eGeMAPS / ComParE / Resemblyzer, resemblyzer raw, permutation tests for eGeMAPS,
and sensitivity analyses (merged speaker groups; audio-derived gender)."""
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))

import sys, os, numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import experiments as E; import extended as X
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
def job(fs,mn,p,s,gn,perm=None,variant='main'):
    import warnings; warnings.filterwarnings("ignore")
    y0=E.y.copy(); g0=dict(E.G); gen=E.gender.copy()
    if variant=='merged':
        sp=E.M.speaker.replace({'S47':'S46','S23':'S28'}).values; E.G={**g0,'speaker':sp}
    if variant=='audiogender':
        gen=gen.copy(); ids=E.M.id.values
        for i in ['lie_047','lie_048','lie_049','lie_052']: gen[ids==i]=0
        gen[ids=='lie_051']=1
    if perm is not None: E.y=np.random.RandomState(1000+perm).permutation(y0)
    try:
        pipe,grid=X.pipes()[mn]; Xm=X.SETS[fs]
        if gn: Xm=np.c_[Xm,gen]; pipe=Pipeline([("gn",E.GenderNorm())]+pipe.steps)
        oof,pred=E.run_cv(pipe,grid,Xm,p,s); r=E.metrics(oof,pred)
    finally: E.y=y0; E.G=g0
    r.update(fs=fs+('_gn' if gn else ''),model=mn,protocol=p,seed=s,perm=-1 if perm is None else perm,variant=variant); return r
if __name__=="__main__":
    J=[]
    for fs in ['egemaps','compare','resemblyzer']:
        for mn in ['SVM-RBF','LogReg']:
            for p in ['random','speaker','trial']:
                for s in range(10):
                    J.append((fs,mn,p,s,True))
                    if fs=='resemblyzer': J.append((fs,mn,p,s,False))
    for p in ['speaker','trial']:
        for gn in [False,True]:
            for i in range(100): J.append(('egemaps','LogReg',p,0,gn,i))
    for s in range(10):
        J.append(('egemaps','LogReg','speaker',s,False,None,'merged')); J.append(('egemaps','SVM-RBF','speaker',s,False,None,'merged'))
        J.append(('egemaps','LogReg','speaker',s,True,None,'audiogender')); J.append(('egemaps','LogReg','trial',s,True,None,'audiogender'))
    print(len(J),'jobs',flush=True); t=time.time()
    out=Parallel(n_jobs=2)(delayed(job)(*j) for j in J)
    pd.DataFrame(out).to_csv('cv_extended2.csv',index=False); print('%.0fs'%(time.time()-t))
