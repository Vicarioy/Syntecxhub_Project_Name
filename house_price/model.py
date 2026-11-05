"""Model construction, training, and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import clean_data


def build_model(features: pd.DataFrame) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(drop="first", handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression()),
        ]
    )


def get_feature_names(model: Pipeline) -> np.ndarray:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def report_coefficients(model: Pipeline, limit: int = 20) -> pd.DataFrame:
    coefficients = model.named_steps["regression"].coef_
    table = pd.DataFrame(
        {"feature": get_feature_names(model), "coefficient": coefficients}
    )
    return table.reindex(
        table["coefficient"].abs().sort_values(ascending=False).index
    ).head(limit)


def train(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    cleaned = clean_data(data, target)
    X = cleaned.drop(columns=[target])
    y = cleaned[target]
    if X.empty:
        raise ValueError("The dataset must contain at least one feature.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = build_model(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "rows": len(cleaned),
        "features": len(X.columns),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    examples = X_test.head(5).copy()
    examples.insert(0, "actual_price", y_test.loc[examples.index])
    examples.insert(1, "predicted_price", model.predict(examples[X.columns]))
    return model, metrics, examples
