"""Exemplo de active learning por amostragem de incerteza em classificação binária."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ActiveLearningResult:
    history: pd.DataFrame
    initial_indices: list[int]
    queried_indices: list[list[int]]
    test_indices: list[int]


def select_uncertain_samples(model, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
    """Retorna posições do pool com probabilidades mais próximas de 0,5."""
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo.")
    if len(X_pool) == 0:
        return np.array([], dtype=int)
    probabilities = model.predict_proba(X_pool)[:, 1]
    return np.argsort(np.abs(probabilities - 0.5), kind="stable")[:batch_size]


def run_active_learning(
    X,
    y,
    *,
    initial_size: int = 30,
    batch_size: int = 15,
    rounds: int = 6,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ActiveLearningResult:
    """Simula um anotador com rótulos conhecidos, mantendo o teste fora das consultas.

    Os rótulos do pool são revelados somente depois da escolha por incerteza.
    Em produção, a consulta seria encaminhada a uma pessoa para anotação.
    As métricas de teste servem para relatar o experimento; não escolhem amostras.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X deve ser uma matriz e y um vetor com o mesmo número de linhas.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Este exemplo exige as duas classes binárias 0 e 1.")
    if rounds < 0 or batch_size < 1 or initial_size < 2:
        raise ValueError("Use rounds >= 0, batch_size >= 1 e initial_size >= 2.")
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    development, test = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
    )
    if initial_size >= len(development):
        raise ValueError("initial_size deve deixar pelo menos uma amostra no pool.")
    labeled, pool = train_test_split(
        development, train_size=initial_size,
        stratify=y[development], random_state=random_state,
    )
    initial = labeled.tolist()
    queries = []
    records = []
    for round_number in range(rounds + 1):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[labeled], y[labeled])
        probabilities = model.predict_proba(X[test])[:, 1]
        records.append({
            "round": round_number,
            "n_labeled": len(labeled),
            "pool_size": len(pool),
            "test_accuracy": accuracy_score(y[test], probabilities >= 0.5),
            "test_roc_auc": roc_auc_score(y[test], probabilities),
        })
        if round_number == rounds or len(pool) == 0:
            break
        positions = select_uncertain_samples(model, X[pool], batch_size)
        selected = pool[positions]
        queries.append(selected.tolist())
        # O acesso aos rótulos destas linhas ocorre no próximo model.fit.
        labeled = np.concatenate([labeled, selected])
        pool = np.delete(pool, positions)

    return ActiveLearningResult(pd.DataFrame(records), initial, queries, test.tolist())
