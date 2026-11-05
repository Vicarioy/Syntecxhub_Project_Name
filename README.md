# House Price Prediction

Train a house-price prediction model with scikit-learn's `LinearRegression`.
The workflow loads a CSV or generated demo data, cleans the target, explores
the features, splits the data into training and test sets, evaluates the model,
prints example predictions, and saves the fitted preprocessing/model pipeline.

## Project layout

```text
.
├── house_price/
│   ├── __init__.py   # Public data and model helpers
│   ├── __main__.py   # Enables `python -m house_price`
│   ├── cli.py        # Command-line arguments and training workflow
│   ├── data.py       # Demo data, CSV loading, target detection, and cleaning
│   └── model.py      # Preprocessing, linear regression, metrics, and coefficients
├── house_price_prediction.py  # Backward-compatible script entry point
└── requirements.txt
```

## Install

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the demo

The demo creates 500 rows of reproducible synthetic housing data; it does not
need an external dataset:

```powershell
python -m house_price --demo
```

The original script entry point works too:

```powershell
python house_price_prediction.py --demo
```

The model is saved to `house_price_model.joblib` in the current directory by
default.

## Train from a CSV

The CSV must have a numeric price column. Common names (`median_house_value`,
`SalePrice`, `sale_price`, `house_price`, and `price`) are detected
automatically. Specify `--target` for other column names:

```powershell
python -m house_price --data housing.csv --target SalePrice
```

Select features explicitly with a comma-separated list and optionally change
the test split and output path:

```powershell
python -m house_price `
  --data housing.csv `
  --target SalePrice `
  --features "OverallQual,GrLivArea,Neighborhood" `
  --test-size 0.25 `
  --output models\house_price_model.joblib
```

Do not include the target column in `--features`. If `--features` is omitted,
all columns other than the target are used.

## Data preparation and model

- Duplicate rows and rows with missing or non-numeric target values are removed.
- Infinite values are treated as missing. Missing numeric and categorical
  feature values are imputed using the training data.
- Numeric features are standardized. Categorical features are one-hot encoded;
  the first category is the reference level and unseen categories are ignored.
- The test-set report includes root mean squared error (RMSE), R², coefficient
  estimates, and up to five actual/predicted examples.
- Numeric feature coefficients are expressed per one standard deviation of
  that feature. Coefficients describe the fitted linear model, not causal
  effects or guaranteed future prices.

## Load the saved model

The joblib file stores the fitted pipeline, target column name, and feature
column names. Provide new rows with the same input columns:

```python
import joblib
import pandas as pd

saved = joblib.load("house_price_model.joblib")
new_houses = pd.DataFrame(
    [{"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr"}]
)
prices = saved["model"].predict(new_houses[saved["features"]])
print(prices)
```

Only load joblib files from trusted sources because loading serialized Python
objects can execute code.
