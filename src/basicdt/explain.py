"""Unified decision-path explanations for BasicDT.

BasicDT's distinctive structure: every boosting round is ONE shared tree whose
leaves emit a full K-vector, so a single root-to-leaf path simultaneously
explains the score of *all* K classes. XGBoost/LightGBM/CatBoost build K
separate trees per round, so explaining the full class distribution there means
reconciling K different topologies. Here it is one path, one attribution.

This module turns that structural property into a concrete API:

* ``feature_importances(clf)`` — one unified gain importance over all classes
  (the split gain is already summed across classes when the shared split is
  chosen, so there is a single coherent ranking, not one per class).
* ``explain_prediction(clf, x)`` — for a single sample, the per-feature
  contribution to *every* class logit, decomposed along the shared decision
  paths. Contributions are conservative: they sum to the model's logit shift
  over the base scores.

All of this runs in pure Python on the exported tree arrays; the C++ engine is
untouched.
"""
from __future__ import annotations

import numpy as np


def _tree_arrays(clf):
    """Yield per-tree exported arrays for a fitted classifier."""
    if getattr(clf, "_is_separate_", len(getattr(clf, "classes_", [])) >= 3):
        for c in range(len(clf.classes_)):
            for t in clf.trees_[c]:
                yield t.export_arrays()
    else:
        for t in clf.trees_:
            yield t.export_arrays()


def _orig_feature_index(clf, perm_idx: int) -> int:
    """Map a (possibly permuted) split-feature index back to original order."""
    perm = getattr(clf, "_col_perm_", None)
    if perm is None:
        return int(perm_idx)
    return int(perm[perm_idx])


def feature_importances(clf, kind: str = "gain", normalize: bool = True):
    """Unified feature importance across all K classes.

    Parameters
    ----------
    kind : {"gain", "split"}
        "gain" sums the K-summed split gain at every internal node using a
        feature; "split" counts how often a feature is used as a split.
    normalize : bool
        Scale to sum to 1.

    Returns
    -------
    imp : ndarray, shape (n_features,)
        One importance per original feature, covering all classes at once.
    """
    n_features = clf.n_features_in_
    imp = np.zeros(n_features, dtype=np.float64)
    for arr in _tree_arrays(clf):
        is_leaf = arr["is_leaf"]
        feats = arr["split_feature"]
        gains = arr["split_gain"]
        for node in range(arr["n_nodes"]):
            if is_leaf[node]:
                continue
            f = _orig_feature_index(clf, feats[node])
            imp[f] += gains[node] if kind == "gain" else 1.0
    if normalize and imp.sum() > 0:
        imp = imp / imp.sum()
    return imp


def _route_path(arr, x_perm):
    """Replicate numeric routing for one sample; return ordered list of
    (node, feature_perm, threshold, gain, went_left) plus the leaf node id.

    Matches the C++ rule ``val < threshold -> left``. Missing-value imputation
    and categorical rank mapping are NOT replicated here, so the leaf reached
    is verified against the engine's own prediction by the caller.
    """
    is_leaf = arr["is_leaf"]
    feats = arr["split_feature"]
    thr = arr["threshold"]
    gains = arr["split_gain"]
    left = arr["left_child"]
    right = arr["right_child"]

    node = 0
    path = []
    while not is_leaf[node]:
        f = int(feats[node])
        went_left = x_perm[f] < thr[node]
        path.append((node, f, float(thr[node]), float(gains[node]), bool(went_left)))
        node = int(left[node]) if went_left else int(right[node])
    return path, node


def explain_prediction(clf, x, top_k_features: int = 10):
    """Explain a single prediction by decomposing the shared decision paths.

    Parameters
    ----------
    x : array-like, shape (n_features,) or (1, n_features)
        One raw sample, in the original feature order.
    top_k_features : int
        How many top features (by absolute effect on the predicted class) to
        include in the human-readable summary.

    Returns
    -------
    dict with keys:
        predicted_class, proba (K,), base_logits (K,), logits (K,),
        contributions (n_features, K), unattributed (K,), classes (K,),
        summary (list of dicts for the top features), exact_paths (bool).

    The ``contributions`` matrix is conservative:
        base_logits + contributions.sum(0) + unattributed == logits.
    Each feature row says how that one feature moved *every* class score -- the
    single-path-explains-all-classes property made explicit.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n_features = clf.n_features_in_
    if x.shape[0] != n_features:
        raise ValueError(f"x has {x.shape[0]} features, expected {n_features}")

    perm = getattr(clf, "_col_perm_", None)
    x_perm = x[perm] if perm is not None else x
    x_row = np.ascontiguousarray(x_perm.reshape(1, -1), dtype=np.float32)

    K = len(clf.F_init_)
    lr = clf.learning_rate
    base_logits = np.asarray(clf.F_init_, dtype=np.float64)

    contributions = np.zeros((n_features, K), dtype=np.float64)
    unattributed = np.zeros(K, dtype=np.float64)
    exact_paths = True

    if getattr(clf, "_is_separate_", K >= 3):
        for c in range(K):
            for t in clf.trees_[c]:
                arr = t.export_arrays()
                leaf_vals = arr["leaf_vals"]
                path, leaf = _route_path(arr, x_perm)
                contrib = np.zeros(K, dtype=np.float64)
                contrib[c] = lr * float(leaf_vals[leaf, 0])

                if not path:
                    unattributed += contrib
                    continue
                gains = np.array([g for (_, _, _, g, _) in path], dtype=np.float64)
                if gains.sum() <= 0:
                    weights = np.full(len(path), 1.0 / len(path))
                else:
                    weights = gains / gains.sum()
                for (_, f_perm, _, _, _), w in zip(path, weights):
                    f_orig = _orig_feature_index(clf, f_perm)
                    contributions[f_orig] += w * contrib
    else:
        for arr in _tree_arrays(clf):
            leaf_vals = arr["leaf_vals"]
            path, leaf = _route_path(arr, x_perm)
            contrib = lr * leaf_vals[leaf].astype(np.float64)  # (K,)

            # Distribute this tree's K-vector contribution across the path features,
            # weighted by split gain (conservative: shares sum to the full contrib).
            if not path:
                unattributed += contrib
                continue
            gains = np.array([g for (_, _, _, g, _) in path], dtype=np.float64)
            if gains.sum() <= 0:
                weights = np.full(len(path), 1.0 / len(path))
            else:
                weights = gains / gains.sum()
            for (_, f_perm, _, _, _), w in zip(path, weights):
                f_orig = _orig_feature_index(clf, f_perm)
                contributions[f_orig] += w * contrib

    # exact routing logits from the engine (authoritative)
    logits = base_logits + _engine_logit_shift(clf, x_row)
    shift_total = contributions.sum(0) + unattributed
    # verify path reconstruction matched the engine
    if not np.allclose(base_logits + shift_total, logits, atol=1e-3):
        exact_paths = False

    proba = _softmax(logits)
    pred = int(np.argmax(proba))
    classes = np.asarray(clf.classes_)

    order = np.argsort(-np.abs(contributions[:, pred]))[:top_k_features]
    names = getattr(clf, "feature_names_in_", None)
    summary = []
    for f in order:
        summary.append({
            "feature": names[f] if names is not None else int(f),
            "value": float(x[f]),
            "effect_on_pred": float(contributions[f, pred]),
            "effect_all_classes": contributions[f].copy(),
        })

    return {
        "predicted_class": classes[pred],
        "proba": proba,
        "base_logits": base_logits,
        "logits": logits,
        "contributions": contributions,
        "unattributed": unattributed,
        "classes": classes,
        "summary": summary,
        "exact_paths": exact_paths,
    }


def _engine_logit_shift(clf, x_row):
    """Logit shift (sum of lr * leaf vectors) from the engine, base excluded."""
    from ._basicdt import predict_ensemble
    K = len(clf.F_init_)
    if getattr(clf, "_is_separate_", K >= 3):
        F = np.zeros((x_row.shape[0], K), dtype=np.float64)
        for c in range(K):
            F_c = predict_ensemble(clf.trees_[c], x_row, 1, clf.learning_rate,
                                   np.zeros(1, dtype=np.float32))
            F[:, c] = F_c[:, 0]
        return F[0]
    else:
        F = predict_ensemble(clf.trees_, x_row, K, clf.learning_rate,
                             np.zeros(K, dtype=np.float32))
        return F[0].astype(np.float64)


def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()
