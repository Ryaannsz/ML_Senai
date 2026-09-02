import numpy as np
import pandas as pd

from module_olist.modeling.evaluate import evaluate_models, find_best_threshold
from module_olist.modeling.split import FEATURES, split_data


def make_data(n_rows: int = 20) -> pd.DataFrame:
    data = pd.DataFrame({feature: 1 for feature in FEATURES}, index=range(n_rows))
    data["customer_state"] = "SP"
    data["order_purchase_timestamp"] = pd.date_range("2024-01-01", periods=n_rows)
    data["is_late"] = np.arange(n_rows) % 2
    return data


def test_split_data_keeps_the_latest_period_as_test() -> None:
    data = make_data()
    data = data.sample(frac=1, random_state=7)

    X_train, X_test, y_train, y_test = split_data(data, test_size=0.2)

    train_dates = pd.to_datetime(data.loc[X_train.index, "order_purchase_timestamp"])
    test_dates = pd.to_datetime(data.loc[X_test.index, "order_purchase_timestamp"])
    assert train_dates.max() < test_dates.min()
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(y_train) + len(y_test) == len(data)


def test_threshold_is_selected_from_validation_predictions() -> None:
    metrics = find_best_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.10, 0.20, 0.80, 0.90]),
    )

    assert metrics["threshold"] == 0.21
    assert metrics["f1"] == 1.0


def test_evaluation_uses_the_threshold_received_from_cv() -> None:
    class DummyModel:
        def predict_proba(self, X):
            probabilities = np.asarray(X["promised_days"], dtype=float)
            return np.column_stack([1 - probabilities, probabilities])

    X_test = pd.DataFrame({"promised_days": [0.1, 0.8]})
    y_test = pd.Series([0, 1])
    results = evaluate_models(
        {"dummy": DummyModel()},
        X_test,
        y_test,
        thresholds={"dummy": 0.75},
    )

    assert results.loc[0, "threshold"] == 0.75
    assert results.loc[0, "f1"] == 1.0
    assert results.loc[0, "accuracy"] == 1.0
    assert results.loc[0, "pr_auc"] == 1.0
