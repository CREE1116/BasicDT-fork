"""Many-class scaling benchmark: BasicDT vs XGBoost vs LightGBM.

Goal: measure whether BasicDT's shared-tree (flat-in-K) design actually
beats the O(K)-trees-per-round competitors as K grows. This is the
'niche verification' run -- accuracy AND fit time vs K.
"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

from basicdt import BasicDTClassifier
import xgboost as xgb
import lightgbm as lgb

N        = 30_000
FEATS    = 30
INFORM   = 15
ESTIM    = 100
DEPTH    = 6
LR       = 0.1
KS       = [10, 50, 100, 200]


def make_data(K, seed=0):
    X, y = make_classification(
        n_samples=N, n_features=FEATS, n_informative=INFORM,
        n_redundant=5, n_classes=K, n_clusters_per_class=1,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def timed_fit(model, Xtr, ytr):
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    return time.perf_counter() - t


def run_basicdt(Xtr, ytr, Xte):
    m = BasicDTClassifier(n_estimators=ESTIM, learning_rate=LR,
                          max_depth=DEPTH, subsample=0.8)
    ft = timed_fit(m, Xtr, ytr)
    return ft, m.predict(Xte)


def run_xgb(Xtr, ytr, Xte, K):
    m = xgb.XGBClassifier(n_estimators=ESTIM, learning_rate=LR,
                          max_depth=DEPTH, subsample=0.8,
                          tree_method="hist", max_bin=256,
                          num_class=K, n_jobs=-1, verbosity=0)
    ft = timed_fit(m, Xtr, ytr)
    return ft, m.predict(Xte)


def run_lgb(Xtr, ytr, Xte):
    m = lgb.LGBMClassifier(n_estimators=ESTIM, learning_rate=LR,
                           max_depth=DEPTH, subsample=0.8,
                           max_bin=255, n_jobs=-1, verbose=-1)
    ft = timed_fit(m, Xtr, ytr)
    return ft, m.predict(Xte)


print(f"N={N} feats={FEATS} estim={ESTIM} depth={DEPTH} lr={LR}")
print(f"{'K':>5} | {'BasicDT acc/fit':>22} | {'XGB acc/fit':>22} | {'LGBM acc/fit':>22}")
print("-" * 85)

for K in KS:
    Xtr, Xte, ytr, yte = make_data(K)
    b_ft, b_pred = run_basicdt(Xtr, ytr, Xte)
    x_ft, x_pred = run_xgb(Xtr, ytr, Xte, K)
    l_ft, l_pred = run_lgb(Xtr, ytr, Xte)
    b_acc = accuracy_score(yte, b_pred)
    x_acc = accuracy_score(yte, x_pred)
    l_acc = accuracy_score(yte, l_pred)
    print(f"{K:>5} | {b_acc:6.4f} / {b_ft:7.1f}s    | "
          f"{x_acc:6.4f} / {x_ft:7.1f}s    | {l_acc:6.4f} / {l_ft:7.1f}s")
