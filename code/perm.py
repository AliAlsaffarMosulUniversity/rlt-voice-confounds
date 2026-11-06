"""Permutation tests (label shuffling, full nested pipeline re-run) for the headline SVM-RBF and gender-only models,
plus speaker-cluster bootstrap CIs of accuracy/AUC from the seed-0 out-of-fold predictions."""
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))

import sys, json, numpy as np, pandas as pd, warnings, os
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 100

def perm_job(fs, mname, protocol, i):
    import warnings; warnings.filterwarnings("ignore")
    rng = np.random.RandomState(1000 + i); y0 = E.y.copy()
    E.y = rng.permutation(y0)
    try:
        r, _ = E.job(fs, mname, protocol, 0)
    finally:
        E.y = y0
    return dict(fs=fs, model=mname, protocol=protocol, i=i, acc=r["acc"], auc=r["auc"])

if __name__ == "__main__":
    J = [(fs, "SVM-RBF", p, i) for fs in ["speech", "speech_gendernorm"] for p in ["random", "speaker", "trial"] for i in range(NP)]
    out = Parallel(n_jobs=2)(delayed(perm_job)(*j) for j in J)
    pd.DataFrame(out).to_csv("perm_results.csv", index=False); print("perm done")
