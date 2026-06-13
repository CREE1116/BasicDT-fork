"""A/B: colsample_bynode regularizer. Same model, vary colsample only.

Uses a high-dim, noisy many-class set (many uninformative features) where a
feature-subsampling regularizer should lift held-out accuracy.
"""
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from basicdt import BasicDTClassifier

X, y = make_classification(
    n_samples=20000, n_features=80, n_informative=18, n_redundant=10,
    n_repeated=0, n_classes=12, n_clusters_per_class=1, flip_y=0.03,
    class_sep=0.8, random_state=0,
)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

print(f"N={len(Xtr)} feats={X.shape[1]} K=12 noisy")
print(f"{'colsample':>10} | {'train acc':>9} | {'test acc':>9} | {'fit s':>7}")
print("-" * 48)

for cs in [1.0, 0.8, 0.6, 0.4, 0.25]:
    clf = BasicDTClassifier(n_estimators=150, learning_rate=0.1, max_depth=6,
                            subsample=0.8, colsample_bynode=cs, random_state=0)
    t = time.perf_counter()
    clf.fit(Xtr, ytr)
    ft = time.perf_counter() - t
    tr = (clf.predict(Xtr) == ytr).mean()
    te = (clf.predict(Xte) == yte).mean()
    print(f"{cs:>10.2f} | {tr:>9.4f} | {te:>9.4f} | {ft:>7.1f}")

# determinism check: colsample=1.0 twice -> identical
a = BasicDTClassifier(n_estimators=30, colsample_bynode=1.0, random_state=0).fit(Xtr, ytr)
b = BasicDTClassifier(n_estimators=30, colsample_bynode=1.0, random_state=0).fit(Xtr, ytr)
same = np.array_equal(a.predict(Xte), b.predict(Xte))
print(f"\ncolsample=1.0 reproducible across runs: {same}")
