from pathlib import Path

from loguru import logger
from models.dataset import create_dataset, load_data, save_dataset
from models.feature import create_features

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.split import split_data
from module_olist.modeling.train import train_models


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

    logger.info("Separando os dados de treino e teste...")
    X_train, X_test, y_train, y_test = split_data(features)

    logger.info("Treinando os modelos...")
    models = train_models(X_train, y_train)

    logger.info("Avaliando os modelos...")
    evaluate_models(models, X_test, y_test)

    logger.success("Pipeline de dados e modelagem concluído com sucesso.")
    return 0


if __name__ == "__main__":
    main()
