
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import sys, numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import experiments as E, extended as X, incremental as I
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score
wit=E.M.role.isin(['witness','victim','prosecutor witness']).values
mask=~wit
def run(model,p,s,perm=None):
    import warnings; warnings.filterwarnings("ignore")
    E.M=E.M[mask].reset_index(drop=True) if len(E.M)==121 else E.M
    E.y=E.y[mask] if len(E.y)==121 else E.y
    E.G={k:(v[mask] if len(v)==121 else v) for k,v in E.G.items()}
    I.EG=X.SETS['egemaps'][mask]; I.g=I.g[mask] if len(I.g)==121 else I.g
    r=I.run(model,p,s,perm); r.pop('oof'); r.pop('pred'); return r
if __name__=="__main__":
    J=[(m,p,s) for m in ["M0","M1"] for p in ["speaker","trial"] for s in range(10)]+[("M1",p,0,i) for p in ["speaker","trial"] for i in range(150)]
    t=time.time(); out=Parallel(n_jobs=2)(delayed(run)(*j) for j in J)
    R=pd.DataFrame(out); R.to_csv('incremental_role.csv',index=False)
    m=R[R.perm==-1]; print(m.groupby(['model','protocol'])[['acc','auc']].mean().round(3))
    for p in ['speaker','trial']:
        o=m[(m.model=='M1')&(m.protocol==p)&(m.seed==0)].iloc[0]; P=R[(R.perm>=0)&(R.protocol==p)]
        print(p,'obs %.3f/%.3f null %.3f/%.3f p_acc %.3f p_auc %.3f'%(o.acc,o.auc,P.acc.mean(),P.auc.mean(),((P.acc>=o.acc).sum()+1)/(len(P)+1),((P.auc>=o.auc).sum()+1)/(len(P)+1)))
    print('%.0fs'%(time.time()-t))
