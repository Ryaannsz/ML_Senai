from pathlib import Path

import pandas as pd
import typer

from module_olist.config import PROCESSED_DATA_DIR
from module_olist.dataset import save_dataset

app = typer.Typer()


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria apenas variáveis disponíveis no momento da aprovação do pedido."""
    data = data.copy()
    data["promised_days"] = (
        (data["order_estimated_delivery_date"] - data["order_approved_at"])
        .dt.total_seconds()
        .div(86_400)
    )
    data["purchase_month"] = data["order_purchase_timestamp"].dt.month
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.dayofweek
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour
    return data


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    """Gera e salva as variáveis de modelagem."""
    data = pd.read_csv(
        input_path,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    save_dataset(create_features(data), output_path)


if __name__ == "__main__":
    app()
