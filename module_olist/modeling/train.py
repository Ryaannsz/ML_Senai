from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from module_olist.modeling.evaluate import find_best_threshold
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)

MODEL_FACTORIES = {
    "Gradient Boosting": create_gradient_boosting_pipeline,
    "XGBoost": create_xgboost_pipeline,
    "LightGBM": create_lightgbm_pipeline,
}


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Executa CV temporal e gera previsões out-of-fold para calibrar thresholds.

    O pré-processador está dentro de cada Pipeline e é ajustado novamente em
    cada fold, evitando que imputação e one-hot usem informação futura.
    """
    if n_splits < 2:
        raise ValueError("n_splits deve ser pelo menos 2.")
    if len(X_train) <= n_splits:
        raise ValueError("Há poucas amostras para a quantidade de folds informada.")

    cv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    thresholds = {}

    for name, factory in MODEL_FACTORIES.items():
        oof_proba = np.full(len(X_train), np.nan, dtype=float)
        fold_auc = []
        fold_pr_auc = []

        for fold, (fit_idx, validation_idx) in enumerate(cv.split(X_train), start=1):
            model = clone(factory())
            model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            probabilities = model.predict_proba(X_train.iloc[validation_idx])[:, 1]
            oof_proba[validation_idx] = probabilities

            fold_pr_auc.append(average_precision_score(
                y_train.iloc[validation_idx], probabilities
            ))

            if y_train.iloc[validation_idx].nunique() > 1:
                fold_auc.append(roc_auc_score(y_train.iloc[validation_idx], probabilities))
            logger.debug("{} - fold {}/{} concluído", name, fold, n_splits)

        valid_oof = ~np.isnan(oof_proba)
        threshold_metrics = find_best_threshold(
            y_train.iloc[valid_oof].to_numpy(), oof_proba[valid_oof]
        )
        thresholds[name] = threshold_metrics["threshold"]
        results.append(
            {
                "model": name,
                "cv_pr_auc_mean": float(np.mean(fold_pr_auc)),
                "cv_roc_auc_mean": float(np.mean(fold_auc)),
                "cv_roc_auc_std": float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
                "oof_roc_auc": roc_auc_score(y_train.iloc[valid_oof], oof_proba[valid_oof]),
                "cv_threshold": threshold_metrics["threshold"],
                "cv_precision": threshold_metrics["precision"],
                "cv_recall": threshold_metrics["recall"],
                "cv_f1": threshold_metrics["f1"],
            }
        )

    return pd.DataFrame(results), thresholds


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, float]]:
    """Valida, calibra thresholds e ajusta os modelos finais no treino."""
    cv_results, thresholds = cross_validate_models(
        X_train,
        y_train,
        n_splits=n_splits,
    )

    trained_models = {}
    for name, factory in MODEL_FACTORIES.items():
        logger.info("Treinando modelo final: {}", name)
        model = factory()
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, cv_results, thresholds
