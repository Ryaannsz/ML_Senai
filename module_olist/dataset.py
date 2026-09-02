from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_data(
    orders_path: Path,
    items_path: Path,
    customers_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega as tabelas brutas usadas na base de modelagem."""
    orders = pd.read_csv(
        orders_path,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)
    return orders, items, customers


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Salva um DataFrame criando o diretório de destino, se necessário."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.success(f"Dataset salvo em {output_path}")


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Seleciona pedidos observáveis e cria o alvo de atraso."""
    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_approved_at"].notna()
    ].copy()

    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8")

    logger.info(f"Pedidos originais: {len(orders):,}")
    logger.info(f"Pedidos no recorte histórico: {len(delivered_orders):,}")
    logger.info(delivered_orders["is_late"].value_counts(dropna=False))
    return delivered_orders


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Agrega os itens para manter uma linha por pedido."""
    items_agg = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    assert items_agg["order_id"].is_unique
    return items_agg


def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Monta uma base com uma linha por pedido entregue."""
    target = create_target(orders)
    items_agg = aggregate_items(items)

    data = target.merge(
        items_agg,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    data = data.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    logger.info(f"Dataset criado com {len(data):,} pedidos e {len(data.columns)} colunas.")
    return data


@app.command()
def main(
    orders_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    items_path: Path = RAW_DATA_DIR / "olist_order_items_dataset.csv",
    customers_path: Path = RAW_DATA_DIR / "olist_customers_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    """Cria e salva o dataset de modelagem a partir dos CSVs brutos."""
    orders, items, customers = load_data(orders_path, items_path, customers_path)
    save_dataset(create_dataset(orders, items, customers), output_path)


if __name__ == "__main__":
    app()
