import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from module_olist.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"


def load_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Carrega o pipeline treinado e seus metadados de inferência."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_keys = {"features", "threshold", "model_name"}
    missing_keys = required_keys.difference(metadata)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Metadados incompletos; campos ausentes: {missing}")
    return model, metadata


def predict(
    data: pd.DataFrame,
    model: Any,
    metadata: dict[str, Any],
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Gera probabilidade e classe prevista para cada pedido.

    O threshold padrão vem dos metadados gerados durante a validação cruzada.
    """
    feature_names = metadata.get("features")
    if not feature_names:
        raise ValueError("Os metadados não informam as features do modelo.")

    missing_features = sorted(set(feature_names).difference(data.columns))
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Features ausentes para inferência: {missing}")

    probability = model.predict_proba(data[feature_names])[:, 1]
    prediction_threshold = float(metadata["threshold"] if threshold is None else threshold)
    if not 0 <= prediction_threshold <= 1:
        raise ValueError("O threshold deve estar entre 0 e 1.")

    predictions = pd.DataFrame(
        {
            "is_late_probability": probability,
            "is_late_prediction": (probability >= prediction_threshold).astype("int8"),
        },
        index=data.index,
    )
    if "order_id" in data.columns:
        predictions.insert(0, "order_id", data["order_id"])
    if "is_late" in data.columns:
        predictions["is_late"] = data["is_late"]

    return predictions


def predict_from_csv(
    features_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Lê features de um CSV, executa a inferência e devolve as previsões."""
    features_path = Path(features_path)
    model, metadata = load_artifacts(model_path, metadata_path)
    features = pd.read_csv(features_path)
    return predict(features, model, metadata, threshold=threshold)
