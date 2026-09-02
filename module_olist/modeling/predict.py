import json
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from module_olist.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "best_model.joblib",
    metadata_path: Path = MODELS_DIR / "best_model_metadata.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
) -> None:
    """Gera previsões usando o pipeline e o threshold salvos no treinamento."""
    logger.info("Carregando modelo e dados para inferência...")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = pd.read_csv(features_path)
    X = features[metadata["features"]]
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(metadata["threshold"])

    predictions = pd.DataFrame(
        {
            "is_late_probability": probabilities,
            "is_late_prediction": (probabilities >= threshold).astype("int8"),
        }
    )
    if "order_id" in features:
        predictions.insert(0, "order_id", features["order_id"])
    if "is_late" in features:
        predictions["is_late"] = features["is_late"]

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    logger.success("Previsões salvas em {}", predictions_path)


if __name__ == "__main__":
    app()
