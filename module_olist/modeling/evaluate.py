from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLDS = np.round(np.arange(0.05, 0.51, 0.01), 2)


def find_best_threshold(y_true, y_proba) -> dict[str, float]:
    """Escolhe o threshold usando somente previsões de validação."""
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def evaluate_models(
    models,
    X_test,
    y_test,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Avalia no teste usando thresholds definidos no treinamento/CV.

    Se nenhum threshold for informado, usa 0.5 fixo; nunca otimiza o
    threshold no teste, evitando vazamento da métrica final.
    """
    thresholds = thresholds or {}
    rows = []
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = float(thresholds.get(name, 0.5))
        y_pred = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        logger.info(f"Modelo: {name}")
        logger.info(f"Threshold definido na validação: {threshold:.2f}")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"PR-AUC: {pr_auc:.3f}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1: {f1:.3f}")
        logger.info(f"ROC-AUC: {roc_auc:.3f}")
        rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "pr_auc": pr_auc,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(rows)
