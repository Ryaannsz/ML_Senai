import numpy as np
import pandas as pd

from module_olist.inference import predict


class DummyModel:
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        probabilities = data["promised_days"].to_numpy(dtype=float)
        return np.column_stack([1 - probabilities, probabilities])


def test_predict_uses_metadata_threshold_and_preserves_identifiers() -> None:
    data = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "promised_days": [0.06, 0.08],
            "is_late": [0, 1],
        }
    )
    metadata = {
        "model_name": "dummy",
        "features": ["promised_days"],
        "threshold": 0.07,
    }

    predictions = predict(data, DummyModel(), metadata)

    assert predictions["is_late_prediction"].tolist() == [0, 1]
    assert predictions["order_id"].tolist() == ["a", "b"]
    assert predictions["is_late"].tolist() == [0, 1]


def test_predict_rejects_missing_features() -> None:
    metadata = {
        "model_name": "dummy",
        "features": ["promised_days"],
        "threshold": 0.07,
    }

    try:
        predict(pd.DataFrame({"total_price": [10.0]}), DummyModel(), metadata)
    except ValueError as error:
        assert "promised_days" in str(error)
    else:
        raise AssertionError("Era esperado erro para feature ausente.")
