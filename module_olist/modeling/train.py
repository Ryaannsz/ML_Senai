import json
from pathlib import Path

import joblib
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
from module_olist.modeling.split import FEATURES, TARGET

MODEL_FACTORIES = {
    "Gradient Boosting": create_gradient_boosting_pipeline,
    "XGBoost": create_xgboost_pipeline,
    "LightGBM": create_lightgbm_pipeline,
}


def select_best_model(
    cv_results: pd.DataFrame,
    *,
    metric: str = "cv_f1",
) -> str:
    """Seleciona o modelo com melhor F1 após aplicar o threshold da CV."""
    if metric not in cv_results.columns:
        raise ValueError(f"Métrica de seleção ausente nos resultados: {metric}")
    if cv_results.empty:
        raise ValueError("Não há resultados de validação cruzada para selecionar um modelo.")
    return str(cv_results.loc[cv_results[metric].idxmax(), "model"])


def save_best_model(
    models: dict[str, Pipeline],
    cv_results: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    output_dir: Path,
    X_final: pd.DataFrame | None = None,
    y_final: pd.Series | None = None,
    selection_metric: str = "cv_f1",
) -> tuple[str, Path, Path]:
    """Salva o melhor pipeline e os metadados necessários para inferência.

    Quando X_final e y_final são informados, o modelo selecionado é reajustado
    com todos os dados rotulados depois que o teste final já foi avaliado.
    """
    best_name = select_best_model(cv_results, metric=selection_metric)
    if best_name not in models:
        raise ValueError(f"Modelo selecionado não foi treinado: {best_name}")

    model = models[best_name]
    training_scope = "train_split"
    if (X_final is None) != (y_final is None):
        raise ValueError("X_final e y_final devem ser informados juntos.")
    if X_final is not None and y_final is not None:
        model = MODEL_FACTORIES[best_name]()
        model.fit(X_final, y_final)
        training_scope = "all_labeled_data_after_test_evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metadata_path = output_dir / "best_model_metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": best_name,
        "selection_metric": selection_metric,
        "threshold": thresholds[best_name],
        "features": FEATURES,
        "target": TARGET,
        "training_scope": training_scope,
        "cv_results": json.loads(cv_results.to_json(orient="records")),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success("Melhor modelo salvo em {}", model_path)
    logger.info("Metadados do modelo salvos em {}", metadata_path)
    return best_name, model_path, metadata_path


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
