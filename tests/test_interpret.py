import numpy as np
import pandas as pd

from module_olist.modeling.interpret import get_feature_names, prepare_data_for_shap
from module_olist.modeling.pipeline import create_xgboost_pipeline


def test_prepare_data_for_shap_uses_transformed_feature_names() -> None:
    X = pd.DataFrame(
        {
            "promised_days": [10.0, 12.0, 8.0],
            "item_count": [1, 2, 1],
            "seller_count": [1, 1, 1],
            "total_price": [50.0, 75.0, 20.0],
            "total_freight": [10.0, 12.0, 8.0],
            "purchase_month": [1, 2, 1],
            "purchase_weekday": [0, 1, 2],
            "purchase_hour": [10, 11, 12],
            "customer_state": ["SP", "RJ", "SP"],
        }
    )
    y = pd.Series([0, 1, 0])
    pipeline = create_xgboost_pipeline().fit(X, y)

    transformed = prepare_data_for_shap(pipeline, X, sample_size=2)

    assert transformed.shape == (2, len(get_feature_names(pipeline)))
    assert transformed.columns.tolist() == get_feature_names(pipeline)
    assert np.isfinite(transformed.to_numpy()).all()
