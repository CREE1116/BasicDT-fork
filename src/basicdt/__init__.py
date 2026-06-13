"""
BasicDT — Fast Histogram-Based Axis-Aligned GBDT.
"""

from .classifier import BasicDTClassifier
from ._basicdt import BasicDTree
from .explain import feature_importances, explain_prediction


def load_model(path: str) -> BasicDTClassifier:
    """Load a model saved with ``clf.save(path)``."""
    return BasicDTClassifier.load(path)


__version__ = "0.1.0"
__all__ = [
    "BasicDTClassifier",
    "BasicDTree",
    "load_model",
    "feature_importances",
    "explain_prediction",
]
