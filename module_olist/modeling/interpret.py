from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _require_shap() -> Any:
    """Importa SHAP somente quando a explicabilidade for solicitada."""
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "A explicabilidade SHAP exige a dependência 'shap'. "
            "Execute 'uv sync' para instalar as dependências do projeto."
        ) from error
    return shap


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Obtém os nomes das colunas após o pré-processamento do pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("O pipeline não possui um pré-processador compatível.")
    return list(preprocessor.get_feature_names_out())


def prepare_data_for_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Aplica somente o pré-processador e prepara uma amostra para o SHAP."""
    if sample_size < 1:
        raise ValueError("sample_size deve ser pelo menos 1.")
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("pipeline deve ser um sklearn Pipeline ajustado.")

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("O pipeline não possui a etapa 'preprocessor'.")

    feature_names = get_feature_names(pipeline)
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=seed)
    transformed = preprocessor.transform(X_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(
        np.asarray(transformed),
        columns=feature_names,
        index=X_sample.index,
    )


def create_explainer(pipeline: Pipeline) -> Any:
    """Cria um TreeExplainer para o classificador de árvores do pipeline."""
    shap = _require_shap()
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise TypeError("O pipeline precisa ter um classificador de árvores na etapa 'classifier'.")
    return shap.TreeExplainer(classifier)


def calculate_shap_values(explainer: Any, X_transformed: pd.DataFrame) -> Any:
    """Calcula os valores SHAP para as features já transformadas."""
    return explainer(X_transformed)
