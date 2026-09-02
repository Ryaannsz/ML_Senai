from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_month",
    "purchase_hour",
    "purchase_weekday",
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",
]

TARGET = "is_late"


def split_data(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: Literal["temporal", "stratified"] = "temporal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa dados e preserva um teste final que não participa do ajuste.

    Para a previsão de atraso, a estratégia temporal é o padrão: os pedidos
    mais recentes ficam no teste, simulando o uso do modelo em produção.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")
    if strategy not in {"temporal", "stratified"}:
        raise ValueError("strategy deve ser 'temporal' ou 'stratified'.")

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    if strategy == "stratified":
        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

    if "order_purchase_timestamp" not in data:
        raise ValueError("A estratégia temporal exige a coluna order_purchase_timestamp.")

    purchase_timestamp = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce")
    if purchase_timestamp.isna().any():
        raise ValueError("order_purchase_timestamp contém datas inválidas ou ausentes.")

    order = np.argsort(purchase_timestamp.to_numpy(), kind="stable")
    split_at = int(len(order) * (1 - test_size))
    if split_at < 1 or split_at >= len(order):
        raise ValueError("A divisão temporal não deixou treino e teste com dados.")

    train_idx, test_idx = order[:split_at], order[split_at:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test
