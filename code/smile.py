import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import opensmile, pandas as pd, numpy as np
from multiprocessing import Pool
ids=pd.read_csv('meta.csv').id.tolist()
def run(fs):
    sm=opensmile.Smile(feature_set=fs, feature_level=opensmile.FeatureLevel.Functionals)
    rows=[]
    for i in ids:
        r=sm.process_file(f'wav/trial_{i}.wav'); r=r.reset_index(drop=True); r.insert(0,'id',i); rows.append(r)
    return pd.concat(rows)
if __name__=='__main__':
    e=run(opensmile.FeatureSet.eGeMAPSv02); e.to_csv('egemaps.csv',index=False); print('egemaps',e.shape)
    c=run(opensmile.FeatureSet.ComParE_2016); c.to_csv('compare.csv',index=False); print('compare',c.shape)
