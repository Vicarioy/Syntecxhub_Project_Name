"""Command-line interface for training the house-price model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import clean_data, find_target, load_data, normalize_columns
from .model import report_coefficients, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean housing data and train a linear-regression price model."
    )
    parser.add_argument("--data", type=Path, help="Path to a housing CSV file.")
    parser.add_argument(
        "--target",
        help="Price column name (auto-detects common names such as SalePrice or price).",
    )
    parser.add_argument(
        "--features",
        help="Optional comma-separated feature columns to use.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("house_price_model.joblib"),
        help="Where to save the fitted model (default: house_price_model.joblib).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train on reproducible synthetic sample data instead of a CSV.",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be greater than 0 and less than 1.")
    if args.demo and args.data:
        parser.error("Use either --demo or --data, not both.")
    return args


def main() -> None:
    args = parse_args()
    try:
        data = normalize_columns(load_data(args.data, args.demo))
        target = find_target(data, args.target)
        cleaned = clean_data(data, target)

        if args.features:
            feature_names = [
                name.strip() for name in args.features.split(",") if name.strip()
            ]
            if len(feature_names) != len(set(feature_names)):
                raise ValueError("--features must not contain duplicate column names.")
            if target in feature_names:
                raise ValueError(
                    f"The target column {target!r} cannot also be used as a feature."
                )
            missing = [name for name in feature_names if name not in cleaned.columns]
            if missing:
                raise ValueError(f"Feature column(s) not found: {', '.join(missing)}")
            if not feature_names:
                raise ValueError("--features must include at least one column name.")
            cleaned = cleaned[feature_names + [target]]

        print(f"Dataset: {len(cleaned)} rows, {len(cleaned.columns) - 1} features")
        print(f"Target: {target}")
        print("\nMissing values by column:")
        print(cleaned.isna().sum().to_string())
        print("\nFeature summary:")
        print(cleaned.drop(columns=[target]).describe(include="all").T.to_string())
        print(f"\nTarget summary:\n{cleaned[target].describe().to_string()}")

        model, metrics, examples = train(cleaned, target, test_size=args.test_size)
        print("\nEvaluation:")
        print(f"  RMSE: {metrics['rmse']:,.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print("\nLargest coefficients (numeric features are standardized):")
        print(report_coefficients(model).to_string(index=False))
        print("\nExample test-set predictions:")
        print(examples.to_string(index=False))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "target": target,
                "features": list(model.feature_names_in_),
            },
            args.output,
        )
        print(f"\nSaved model to: {args.output.resolve()}")
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
