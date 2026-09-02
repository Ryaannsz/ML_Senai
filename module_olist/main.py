from pathlib import Path

from loguru import logger

from module_olist.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import create_dataset, load_data, save_dataset
from module_olist.features import create_features
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.split import FEATURES, split_data
from module_olist.modeling.train import save_best_model, train_models


def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> int:
    """Executa a preparação dos dados, o treino e a avaliação dos modelos."""
    logger.info("Carregando dados brutos da Olist...")
    orders, items, customers = load_data(orders_path, items_path, customers_path)

    dataset = create_dataset(orders, items, customers)
    save_dataset(dataset, dataset_path)

    features = create_features(dataset)
    save_dataset(features, features_path)

    logger.info("Separando temporalmente os dados de treino e teste...")
    X_train, X_test, y_train, y_test = split_data(features, strategy="temporal")

    logger.info("Treinando os modelos com validação cruzada temporal...")
    models, cv_results, thresholds = train_models(X_train, y_train, n_splits=5)
    logger.info("Resultados de validação cruzada:\n{}", cv_results.to_string(index=False))

    logger.info("Avaliando os modelos no período de teste intocado...")
    evaluate_models(models, X_test, y_test, thresholds=thresholds)

    best_name, model_path, metadata_path = save_best_model(
        models,
        cv_results,
        thresholds,
        output_dir=MODELS_DIR,
        X_final=features[FEATURES],
        y_final=features["is_late"],
    )
    logger.success(
        "Modelo selecionado pela CV: {} | pipeline: {} | metadados: {}",
        best_name,
        model_path,
        metadata_path,
    )

    logger.success("Pipeline de dados e modelagem concluído com sucesso.")
    return 0


if __name__ == "__main__":
    main()
