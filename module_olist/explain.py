from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from module_olist.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from module_olist.inference import load_artifacts
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)

app = typer.Typer(help="Gera explicações SHAP para o modelo de árvores salvo.")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "best_model.joblib",
    metadata_path: Path = MODELS_DIR / "best_model_metadata.json",
    figures_dir: Path = FIGURES_DIR,
    sample_size: int = typer.Option(
        2000,
        min=1,
        help="Número máximo de pedidos usados nos gráficos SHAP.",
    ),
    seed: int = typer.Option(42, help="Semente para selecionar a amostra."),
) -> None:
    """Gera gráficos global, beeswarm e waterfall em reports/figures."""
    logger.info("Iniciando análise SHAP...")
    model, metadata = load_artifacts(model_path, metadata_path)

    features = pd.read_csv(features_path)
    X = features[metadata["features"]]
    X_shap = prepare_data_for_shap(model, X, sample_size=sample_size, seed=seed)
    explainer = create_explainer(model)
    shap_values = calculate_shap_values(explainer, X_shap)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # contribuição para a classe atraso


    figures_dir.mkdir(parents=True, exist_ok=True)
    shap = __import__("shap")

    logger.info("Gerando gráfico SHAP de importância global...")
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    bar_path = figures_dir / "shap_feature_importance.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info("Gerando gráfico SHAP beeswarm...")
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    beeswarm_path = figures_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    plt.close()

    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    waterfall_path = figures_dir / "shap_waterfall.png"
    plt.savefig(waterfall_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.success("Gráficos SHAP salvos em {}", figures_dir)


if __name__ == "__main__":
    app()
