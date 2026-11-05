"""Housing dataset generation, loading, and cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CANDIDATES = (
    "median_house_value",
    "saleprice",
    "sale_price",
    "house_price",
    "price",
)


def create_demo_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Create reproducible, synthetic housing data for trying the workflow."""
    rng = np.random.default_rng(random_state)
    area = rng.normal(1_800, 550, rows).clip(500, 5_000)
    bedrooms = rng.integers(1, 7, rows)
    age = rng.integers(0, 100, rows)
    distance = rng.uniform(0.2, 30, rows)
    neighborhood = rng.choice(["urban", "suburban", "rural"], rows)
    neighborhood_effect = (
        pd.Series(neighborhood)
        .map({"urban": 65_000, "suburban": 30_000, "rural": 0})
        .to_numpy()
    )
    price = (
        45_000
        + area * 190
        + bedrooms * 12_000
        - age * 700
        - distance * 1_100
        + neighborhood_effect
        + rng.normal(0, 25_000, rows)
    )
    return pd.DataFrame(
        {
            "area_sqft": area.round(0),
            "bedrooms": bedrooms,
            "home_age_years": age,
            "distance_to_city_miles": distance.round(1),
            "neighborhood": neighborhood,
            "price": price.round(0),
        }
    )


def load_data(data_path: Path | None, demo: bool) -> pd.DataFrame:
    if demo:
        return create_demo_data()
    if data_path is None:
        raise ValueError("Provide a CSV with --data, or use --demo to try sample data.")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming whitespace.")
    return data


def find_target(data: pd.DataFrame, requested_target: str | None) -> str:
    data = normalize_columns(data)
    if requested_target:
        requested_target = requested_target.strip()
        if requested_target not in data.columns:
            raise ValueError(
                f"Target column {requested_target!r} was not found. "
                f"Available columns: {', '.join(map(str, data.columns))}"
            )
        return requested_target

    column_lookup = {str(column).strip().lower(): column for column in data.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in column_lookup:
            return column_lookup[candidate]
    raise ValueError(
        "Could not identify the price column. Pass its name with --target. "
        f"Available columns: {', '.join(map(str, data.columns))}"
    )


def clean_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError("The dataset has no rows.")
    data = normalize_columns(data)
    target = target.strip()
    if target not in data.columns:
        raise ValueError(
            f"Target column {target!r} was not found after cleaning headers."
        )

    data = data.drop_duplicates()
    data = data.replace([np.inf, -np.inf], np.nan)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    if data.empty:
        raise ValueError(f"No usable rows remain after cleaning target {target!r}.")

    features = data.drop(columns=[target])
    all_missing_columns = features.columns[features.isna().all()].tolist()
    if all_missing_columns:
        data = data.drop(columns=all_missing_columns)
    if len(data.columns) == 1:
        raise ValueError("No usable feature columns remain after cleaning.")
    return data
