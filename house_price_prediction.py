"""Entry point and backward-compatible exports for house-price prediction."""

from house_price import (
    TARGET_CANDIDATES,
    build_model,
    clean_data,
    create_demo_data,
    find_target,
    get_feature_names,
    load_data,
    normalize_columns,
    report_coefficients,
    train,
)
from house_price.cli import main, parse_args


if __name__ == "__main__":
    main()
