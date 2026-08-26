from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERICAL_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
]


def create_preprocessor() -> ColumnTransformer:
    """
    Cria o pré-processador para o pipeline de modelagem.
    """
    return ColumnTransformer(
        transformers=[
            # Nas colunas numéricas
            ("numeric", "passthrough", NUMERICAL_FEATURES),
            # Nas colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
    )


def create_gradient_boosting_pipeline() -> Pipeline:

    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100,  # números de árvores
        learning_rate=0.1,  # taxa de aprendizado
        max_depth=3,  # profundidade máxima das árvores
        random_state=42,  # semente para reprodutibilidade
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def create_xgboost_pipeline() -> Pipeline:

    preprocessor = create_preprocessor()

    model = XGBClassifier(
        n_estimators=100,  # números de árvores
        learning_rate=0.1,  # taxa de aprendizado
        max_depth=3,  # profundidade máxima das árvores
        random_state=42,  # semente para reprodutibilidade
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def create_lightgbm_pipeline() -> Pipeline:

    preprocessor = create_preprocessor()

    model = LGBMClassifier(
        n_estimators=100,  # números de árvores
        learning_rate=0.1,  # taxa de aprendizado
        max_depth=3,  # profundidade máxima das árvores
        random_state=42,  # semente para reprodutibilidade
        verbosity=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
