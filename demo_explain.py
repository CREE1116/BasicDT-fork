"""Smoke + correctness test for the unified-path explainer."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from basicdt import BasicDTClassifier

X, y = make_classification(n_samples=4000, n_features=20, n_informative=12,
                           n_redundant=3, n_classes=8, n_clusters_per_class=1,
                           random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

clf = BasicDTClassifier(n_estimators=60, learning_rate=0.1, max_depth=5,
                        subsample=0.9, random_state=0)
clf.fit(Xtr, ytr)
acc = (clf.predict(Xte) == yte).mean()
print(f"K=8 test acc: {acc:.4f}  trees: {clf.get_n_trees()}")

# --- unified gain importance (one ranking for all 8 classes) ---
imp = clf.feature_importances_
print(f"\nfeature_importances_ shape: {imp.shape}  sum: {imp.sum():.4f}")
top = np.argsort(-imp)[:5]
print("top-5 features (unified gain):", [(int(f), round(float(imp[f]), 3)) for f in top])

# --- explain one prediction: one path set, all K classes ---
ex = clf.explain(Xte[0])
print(f"\npredicted class: {ex['predicted_class']}  proba_max: {ex['proba'].max():.3f}")
print(f"exact_paths (path recon matched engine): {ex['exact_paths']}")

# conservativeness: base + contributions.sum + unattributed == logits
recon = ex["base_logits"] + ex["contributions"].sum(0) + ex["unattributed"]
max_err = np.abs(recon - ex["logits"]).max()
print(f"conservativeness max logit error: {max_err:.2e}  (should be ~0)")

# softmax of reconstructed logits == predict_proba
from numpy import exp
p = exp(ex["logits"] - ex["logits"].max()); p /= p.sum()
ref = clf.predict_proba(Xte[:1])[0]
print(f"proba match vs predict_proba: {np.abs(p - ref).max():.2e}")

print("\ntop feature effects on predicted class (and across all classes):")
for s in ex["summary"][:4]:
    eff = s["effect_all_classes"]
    print(f"  feat {s['feature']:>2} = {s['value']:+.2f} | "
          f"pred-effect {s['effect_on_pred']:+.3f} | "
          f"class-range [{eff.min():+.3f}, {eff.max():+.3f}]")
