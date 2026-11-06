import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import numpy as np, pandas as pd, warnings, json, sys, time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
D = ""
M = pd.read_csv(D + "meta.csv").merge(pd.read_csv(D + "features.csv"), on="id")
y = (M.label == "lie").astype(int).values
G = {"speaker": M.speaker.values, "trial": M.trial.values}
gender = (M.gender == "F").astype(int).values
CH = [c for c in M.columns if c.startswith("ch_")]
SPEECH = [c for c in M.columns[M.columns.get_loc("f0_mean"):] if c not in CH]
print("speech feats", len(SPEECH), "channel feats", len(CH))

class GenderNorm(BaseEstimator, TransformerMixin):
    """z-score each feature within gender using training statistics; last column of X is gender (dropped)."""
    def fit(self, X, y=None):
        g = X[:, -1]; Z = X[:, :-1]
        self.s = {k: (Z[g == k].mean(0), Z[g == k].std(0) + 1e-9) for k in (0, 1)}; return self
    def transform(self, X):
        g = X[:, -1]; Z = X[:, :-1].copy()
        for k, (mu, sd) in self.s.items(): Z[g == k] = (Z[g == k] - mu) / sd
        return Z

def models(k):
    sel = [("sc", StandardScaler()), ("kb", SelectKBest(f_classif, k=k))]
    return {
        "SVM-RBF": (Pipeline(sel + [("m", SVC(probability=False))]), {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.01, 0.1]}),
        "LogReg": (Pipeline(sel + [("m", LogisticRegression(max_iter=3000))]), {"m__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (Pipeline(sel + [("m", RandomForestClassifier(300, random_state=0))]), {"m__max_depth": [3, None]}),
        "GradBoost": (Pipeline(sel + [("m", GradientBoostingClassifier(random_state=0))]), {"m__n_estimators": [50, 150], "m__max_depth": [2, 3]}),
        "MLP": (Pipeline(sel + [("m", MLPClassifier((32,), max_iter=2000, random_state=0))]), {"m__alpha": [1e-3, 1e-1, 1]}),
    }

def score_fn(est, X):
    return est.decision_function(X) if hasattr(est, "decision_function") else est.predict_proba(X)[:, 1]

def run_cv(pipe, grid, X, protocol, seed):
    """nested CV; inner CV is grouped whenever the outer CV is grouped."""
    oof = np.zeros(len(y)); pred = np.zeros(len(y), int)
    if protocol == "random":
        outer = StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)
    else:
        g = G[protocol]; nf = 10 if protocol == "speaker" else 8
        outer = StratifiedGroupKFold(nf, shuffle=True, random_state=seed).split(X, y, g)
    for tr, te in outer:
        if protocol == "random": inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        else: inner = list(GroupKFold(5).split(X[tr], y[tr], G[protocol][tr]))
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="accuracy", n_jobs=1).fit(X[tr], y[tr])
        oof[te] = score_fn(gs.best_estimator_, X[te]); pred[te] = gs.predict(X[te])
    return oof, pred

def metrics(oof, pred):
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, oof))

def job(fs, mname, protocol, seed):
    if fs == "speech": X = M[SPEECH].values; k = 20
    elif fs == "speech_gendernorm": X = np.c_[M[SPEECH].values, gender]; k = 20
    elif fs == "channel": X = M[CH].values; k = "all"
    elif fs == "f0_only": X = M[["f0_mean"]].values; k = "all"
    elif fs == "gender_only": X = gender[:, None].astype(float); k = "all"
    pipe, grid = models(k)[mname]
    if fs == "speech_gendernorm": pipe = Pipeline([("gn", GenderNorm())] + pipe.steps)
    oof, pred = run_cv(pipe, grid, X, protocol, seed)
    r = metrics(oof, pred); r.update(fs=fs, model=mname, protocol=protocol, seed=seed)
    return r, (fs, mname, protocol, seed, oof.tolist(), pred.tolist())

if __name__ == "__main__":
    t = time.time(); J = []
    for p in ["random", "speaker", "trial"]:
        for s in range(10):
            for mn in ["SVM-RBF", "LogReg", "RandomForest", "GradBoost", "MLP"]:
                J.append(("speech", mn, p, s))
            for fs in ["speech_gendernorm"]: J.append((fs, "SVM-RBF", p, s)); J.append((fs, "LogReg", p, s))
            for fs in ["channel", "f0_only", "gender_only"]: J.append((fs, "LogReg", p, s)); J.append((fs, "SVM-RBF", p, s))
    print(len(J), "jobs", flush=True)
    out = Parallel(n_jobs=2, verbose=0)(delayed(job)(*j) for j in J)
    pd.DataFrame([o[0] for o in out]).to_csv(D + "cv_results.csv", index=False)
    json.dump([o[1] for o in out], open(D + "oof.json", "w"))
    print("done %.0fs" % (time.time() - t))
