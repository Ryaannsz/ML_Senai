from pathlib import Path

import typer

from module_olist.config import MODELS_DIR, PROCESSED_DATA_DIR
from module_olist.inference import predict_from_csv

app = typer.Typer(help="Gera previsões usando o melhor modelo salvo.")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "best_model.joblib",
    metadata_path: Path = MODELS_DIR / "best_model_metadata.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    threshold: float | None = typer.Option(
        None,
        help="Threshold opcional; por padrão usa o threshold salvo na CV.",
    ),
) -> None:
    """Lê um CSV de features e salva probabilidades e classes previstas."""
    predictions = predict_from_csv(
        features_path,
        model_path,
        metadata_path,
        threshold=threshold,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    typer.echo(f"Previsões salvas em: {predictions_path}")


if __name__ == "__main__":
    app()
