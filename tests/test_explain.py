import numpy as np
from sklearn.datasets import make_classification
from basicdt import BasicDTClassifier


def _fit(K=6, seed=0):
    X, y = make_classification(
        n_samples=1500, n_features=15, n_informative=10, n_redundant=2,
        n_classes=K, n_clusters_per_class=1, random_state=seed,
    )
    clf = BasicDTClassifier(n_estimators=40, learning_rate=0.1, max_depth=4,
                            subsample=1.0, random_state=seed)
    clf.fit(X, y)
    return clf, X, y, K


def test_feature_importances_unified():
    clf, X, y, K = _fit()
    imp = clf.feature_importances_
    assert imp.shape == (X.shape[1],)          # one vector for all K classes
    assert np.isclose(imp.sum(), 1.0, atol=1e-6)
    assert (imp >= 0).all()


def test_explain_is_conservative_and_exact():
    clf, X, y, K = _fit()
    ex = clf.explain(X[0])
    # contributions matrix covers every class
    assert ex["contributions"].shape == (X.shape[1], K)
    # numeric-only data -> path reconstruction must match the engine exactly
    assert ex["exact_paths"] is True
    # conservativeness: base + contributions + unattributed == logits
    recon = ex["base_logits"] + ex["contributions"].sum(0) + ex["unattributed"]
    assert np.allclose(recon, ex["logits"], atol=1e-4)
    # softmax(logits) reproduces predict_proba
    p = np.exp(ex["logits"] - ex["logits"].max()); p /= p.sum()
    ref = clf.predict_proba(X[:1])[0]
    assert np.allclose(p, ref, atol=1e-4)
    # predicted class agrees with predict
    assert ex["predicted_class"] == clf.predict(X[:1])[0]
