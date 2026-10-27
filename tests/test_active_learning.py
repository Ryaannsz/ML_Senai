import numpy as np
import pytest
from sklearn.datasets import make_classification

from module_olist.active_learning import run_active_learning, select_uncertain_samples


def test_queries_use_uncertainty_rather_than_high_confidence():
    class Model:
        def predict_proba(self, X):
            p = np.array([0.99, 0.51, 0.1, 0.49])
            return np.column_stack([1 - p, p])

    assert select_uncertain_samples(Model(), np.zeros((4, 2)), 2).tolist() == [1, 3]


def test_queries_never_reuse_labeled_examples_or_touch_holdout():
    X, y = make_classification(n_samples=160, n_features=8, random_state=7)
    result = run_active_learning(X, y, initial_size=20, batch_size=11, rounds=4)
    selected = result.initial_indices + sum(result.queried_indices, [])
    assert len(selected) == len(set(selected)) == 64
    assert set(selected).isdisjoint(result.test_indices)
    assert result.history.n_labeled.tolist() == [20, 31, 42, 53, 64]
    assert result.history.test_roc_auc.between(0, 1).all()


def test_pool_exhaustion_stops_queries_and_preserves_budget():
    X, y = make_classification(n_samples=80, n_features=6, random_state=42)
    result = run_active_learning(X, y, initial_size=20, batch_size=100, rounds=5)
    assert result.history.n_labeled.tolist() == [20, 60]
    assert result.history.pool_size.tolist() == [40, 0]
    assert len(result.queried_indices) == 1


def test_multiclass_input_is_rejected():
    with pytest.raises(ValueError, match="binárias"):
        run_active_learning(np.zeros((30, 2)), np.arange(30) % 3)
