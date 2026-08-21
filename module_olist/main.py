from pathlib import Path

from loguru import logger
from models.dataset import create_dataset, load_data, save_dataset
from models.feature import create_features

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> int:
    """Executa a preparação dos dados brutos e a criação das features."""
    logger.info("Carregando dados brutos da Olist...")
    orders, items, customers = load_data(orders_path, items_path, customers_path)

    dataset = create_dataset(orders, items, customers)
    save_dataset(dataset, dataset_path)

    features = create_features(dataset)
    save_dataset(features, features_path)

    logger.success("Pipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    main()
