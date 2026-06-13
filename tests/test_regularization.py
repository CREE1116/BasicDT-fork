import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from basicdt import BasicDTClassifier

def test_nan_sparsity_aware_routing():
    # Construct a dataset where missing values have a distinct pattern
    rng = np.random.default_rng(42)
    N = 1000
    X, y = make_classification(n_samples=N, n_features=10, n_classes=2, random_state=42)
    
    # Introduce NaNs in the most predictive feature for a subset of samples
    X_nan = X.copy()
    predictive_feat = 0
    # For samples where y is 0, make 50% of feature 0 NaN
    mask_y0 = (y == 0)
    nan_indices = rng.choice(np.where(mask_y0)[0], size=int(0.5 * mask_y0.sum()), replace=False)
    X_nan[nan_indices, predictive_feat] = np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(X_nan, y, test_size=0.2, random_state=42)
    
    # Fit model with NaN support
    clf = BasicDTClassifier(n_estimators=10, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Verify predictions run and output finite probabilities
    probas = clf.predict_proba(X_test)
    assert not np.isnan(probas).any()
    assert np.allclose(probas.sum(axis=1), 1.0)
    
    # Verify accuracy is reasonable
    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    assert acc > 0.75

def test_l1_regularization_shrinkage():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    
    # Fit with no L1
    clf_no_l1 = BasicDTClassifier(n_estimators=5, max_depth=3, reg_alpha=0.0, random_state=42)
    clf_no_l1.fit(X, y)
    
    # Fit with huge L1
    clf_huge_l1 = BasicDTClassifier(n_estimators=5, max_depth=3, reg_alpha=100.0, random_state=42)
    clf_huge_l1.fit(X, y)
    
    # Extract leaf values from the first tree
    tree_no_l1 = clf_no_l1.trees_[0].export_arrays()
    tree_huge_l1 = clf_huge_l1.trees_[0].export_arrays()
    
    # Leaf weights with huge L1 should be shrunk significantly closer to 0 than with no L1
    no_l1_vals = np.abs(tree_no_l1["leaf_vals"])
    huge_l1_vals = np.abs(tree_huge_l1["leaf_vals"])
    
    # Leaves with non-zero samples (which are actually reached or have leaf values)
    # should be zeroed or heavily shrunk
    assert huge_l1_vals.max() < no_l1_vals.max()
    # At huge L1, most leaf outputs should be exactly or extremely close to 0
    assert np.allclose(huge_l1_vals, 0.0, atol=1e-3)

def test_gamma_pruning():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Fit with gamma = 0.0 (default)
    clf_no_prune = BasicDTClassifier(n_estimators=5, max_depth=6, gamma=0.0, random_state=42)
    clf_no_prune.fit(X, y)
    
    # Fit with high gamma (e.g. 5.0)
    clf_high_prune = BasicDTClassifier(n_estimators=5, max_depth=6, gamma=5.0, random_state=42)
    clf_high_prune.fit(X, y)
    
    # Total nodes across all trees should be smaller for high gamma due to split pruning
    nodes_no_prune = sum(t.export_arrays()["n_nodes"] for t in clf_no_prune.trees_)
    nodes_high_prune = sum(t.export_arrays()["n_nodes"] for t in clf_high_prune.trees_)
    
    assert nodes_high_prune < nodes_no_prune

def test_max_leaves_constraint():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Fit with max_leaves = 4
    clf_4 = BasicDTClassifier(n_estimators=5, max_depth=6, max_leaves=4, random_state=42)
    clf_4.fit(X, y)
    
    # Each tree should have at most 4 leaves, which means at most 2 * 4 - 1 = 7 nodes
    for t in clf_4.trees_:
        meta = t.export_arrays()
        assert meta["n_nodes"] <= 7
        is_leaf = meta["is_leaf"]
        num_leaves = is_leaf.sum()
        assert num_leaves <= 4

    # Fit with max_leaves = 8
    clf_8 = BasicDTClassifier(n_estimators=5, max_depth=6, max_leaves=8, random_state=42)
    clf_8.fit(X, y)
    
    # Each tree should have at most 8 leaves, which means at most 2 * 8 - 1 = 15 nodes
    for t in clf_8.trees_:
        meta = t.export_arrays()
        assert meta["n_nodes"] <= 15
        is_leaf = meta["is_leaf"]
        num_leaves = is_leaf.sum()
        assert num_leaves <= 8


def test_multiclass_separate_trees():
    # Construct a multiclass dataset with K=3 classes
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit (separate trees are default for K>=3)
    clf = BasicDTClassifier(
        n_estimators=10, max_depth=4, learning_rate=0.1,
        random_state=42, verbose=True
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Verify trees_ structure: list of K lists of trees
    assert isinstance(clf.trees_, list)
    assert len(clf.trees_) == 3  # K=3
    for c in range(3):
        assert isinstance(clf.trees_[c], list)
        assert len(clf.trees_[c]) > 0
        for t in clf.trees_[c]:
            assert t._K == 1  # each tree is a single-class tree
            
    # Verify predict_proba outputs are correct shape and sum to 1
    probas = clf.predict_proba(X_test)
    assert probas.shape == (len(y_test), 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    
    # Verify predictions are valid class indices
    preds = clf.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert set(preds).issubset({0, 1, 2})
    
    # Check that accuracy is reasonable
    acc = (preds == y_test).mean()
    assert acc > 0.6
