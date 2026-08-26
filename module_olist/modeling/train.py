import pandas as pd

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Treinando modelo: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
