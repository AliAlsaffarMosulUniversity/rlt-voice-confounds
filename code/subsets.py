import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import experiments as E
full=(E.M.copy(),E.y.copy(),dict(E.G),E.gender.copy())
def setsub(mask):
    E.M=full[0][mask].reset_index(drop=True); E.y=full[1][mask]; E.G={k:v[mask] for k,v in full[2].items()}; E.gender=full[3][mask]
rows=[]
wit=full[0].role.isin(['witness','victim','prosecutor witness']).values
for name,mask in [('female',full[0].gender.values=='F'),('non_witness',~wit),('female_non_witness',(full[0].gender.values=='F')&~wit)]:
    setsub(mask); maj=max(E.y.mean(),1-E.y.mean())
    for p in ['random','speaker']:
        for s in range(10):
            for mn in ['SVM-RBF','LogReg']:
                r,_=E.job('speech',mn,p,s); r.update(subset=name,n=int(mask.sum()),nspk=len(set(E.G['speaker'])),majority=maj); rows.append(r)
    print(name,mask.sum(),flush=True)
pd.DataFrame(rows).to_csv('subset_results.csv',index=False)
r=pd.DataFrame(rows); print(r.groupby(['subset','model','protocol'])[['acc','auc','majority','n','nspk']].mean().round(3))
