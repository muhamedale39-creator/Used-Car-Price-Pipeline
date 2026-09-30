# Used-Car Price Prediction Pipeline

In this project I built a machine learning pipeline that takes a messy, raw used-car dataset from Kaggle, cleans it and engineers features with Pandas, explores it with charts, and predicts car prices with a tuned ensemble of Random Forest and XGBoost regressors.

I split the work into two standalone scripts instead of a notebook:

- `main.py` cleans the data, creates features, caps outliers, and generates EDA charts
- `train.py` encodes the text columns, tunes two models with grid search, combines them into a voting ensemble, and prints the evaluation scores

```
data.csv ──► main.py ──► cleaned_cars_data.csv ──► train.py ──► R², MAE, RMSE
 (raw)     cleaning, features,      (model-ready)     target encoding, 80/20 split,
           outlier capping, EDA                        GridSearchCV (RF + XGBoost),
                │                                      VotingRegressor
                └──► images/ (EDA charts)
```

## How I Cleaned the Data (`main.py`)

**1. Numeric parsing.** `price` and `milage` were stored as text with symbols and commas. I used a regex to strip everything except digits and decimal points, then converted them to numbers.

**2. Regex feature extraction from `engine`.** The raw `engine` column is free text (e.g. `300.0HP 3.0L V6 Cylinder Engine`). I used regex to pull out two numeric columns, `hp` (horsepower) and `liters` (engine size), then dropped the original column.

**3. Three-tier missing value imputation for `hp` and `liters`.** Instead of a single global average, I fill missing values with a fallback chain built on `groupby`:

1. Median of the same **brand and model**
2. Median of the same **brand**
3. **Global** median

**4. Categorical and binary cleanup.**
- I treat `fuel_type` values outside a valid list (Gasoline, Hybrid, E85 Flex Fuel, Diesel, Plug-In Hybrid, Electric) as missing.
- I turn `accident` into `1` if an accident was reported, otherwise `0`.
- I turn `clean_title` into `1` for `Yes`, otherwise `0`.
- I convert placeholder dash values to missing.

**5. Brand and model name fixes.** I repaired truncated brand names (`Land` → `Land Rover`, `Aston` → `Aston Martin`, `Alfa` → `Alfa Romeo`) and removed the leftover `Rover ` / `Martin ` prefixes from model names.

**6. New features.** I created two features and then dropped `model_year`:
- `vehicle_age` = 2026 − `model_year`
- `mileage_per_year` = `milage` / (`vehicle_age` + 1)

**7. Text normalization.** I lowercase and trim `brand`, `model`, `fuel_type`, `transmission`, `ext_col`, and `int_col`, and strip trailing periods. Any remaining missing values become the category `missing`.

**8. Outlier capping.** I cap `price` with the IQR rule: values above Q3 + 1.5 × IQR are set to that upper limit, and values below Q1 − 1.5 × IQR are raised to the lower limit (never below 0).

**9. Exploratory charts.** I save three plots to `images/`:

| File | Chart |
| --- | --- |
| `price_distribution.png` | Distribution of car prices |
| `mileage_vs_price.png` | Mileage vs price scatter plot |
| `top_10_brands.png` | Ten most common brands |

I write the cleaned dataset to `cleaned_cars_data.csv`.

## How I Trained the Models (`train.py`)

**1. Split.** I separate the cleaned data into features (everything except `price`) and the target (`price`), then split it **80% train / 20% test** with `random_state=1`.

**2. Target encoding.** I convert the text columns `brand`, `model`, `fuel_type`, `transmission`, `ext_col`, and `int_col` to numbers with scikit-learn's `TargetEncoder` (continuous target) inside a `ColumnTransformer`. Each category is replaced by a smoothed average of the price for that category, and the remaining numeric columns pass through unchanged. I fit the encoder on the training set only, so no test-set information leaks into the features.

**3. Random Forest with grid search.** I tune a `RandomForestRegressor` with `GridSearchCV`, using 5-fold cross-validation (shuffled, `random_state=1`) and R² as the scoring metric.

| Parameter | Values searched |
| --- | --- |
| `n_estimators` | 100, 150, 200 |
| `max_depth` | 5, 7, 10, 15 |

**4. XGBoost with grid search.** I tune an `XGBRegressor` the same way (5-fold CV, R² scoring).

| Parameter | Values searched |
| --- | --- |
| `n_estimators` | 150, 200, 300 |
| `learning_rate` | 0.1, 0.4, 0.6 |
| `max_depth` | 4, 6, 10, 15 |

**5. Voting ensemble.** I combine the best Random Forest and the best XGBoost model from the searches in a `VotingRegressor`, which averages their predictions. I fit the ensemble on the training set and evaluate it on the held-out test set.

### Evaluation Metrics

`train.py` prints these scores for the ensemble:

| Metric | Score |
| --- | --- |
| **R² score** | 0.9083 (90.8% of data variance explained)|
| **MAE** | $5586.37 |
| **RMSE** | $8605.27 |

## Project Structure

| Path | Description |
| --- | --- |
| `main.py` | Cleaning, feature engineering, outlier capping, EDA charts |
| `train.py` | Target encoding, grid search, ensemble training, evaluation |
| `data.csv` | Raw Kaggle dataset |
| `cleaned_cars_data.csv` | Model-ready output of `main.py` |
| `images/` | EDA charts generated by `main.py` |
| `requirements.txt` | Python dependencies |

## How to Run It Locally

**1. Clone the repository**

```bash
git clone https://github.com/muhamedale39-creator/Used-Car-Price-Pipeline.git
cd Used-Car-Price-Pipeline
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

`train.py` uses `TargetEncoder` and imports `root_mean_squared_error`, so it needs scikit-learn 1.4 or newer, plus the `xgboost` package.

**3. Clean the data and generate charts**

```bash
python main.py
```

The charts open in pop-up windows first. Close each one to let the script continue and save the files to `images/`.

**4. Train and evaluate the models**

```bash
python train.py
```

The grid searches fit 60 Random Forest models (12 combinations × 5 folds) and 180 XGBoost models (36 combinations × 5 folds), so this step can take a while.

## Tech Stack

Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn

## What I Plan to Improve

- Print the tuned Random Forest and XGBoost scores and the best parameters found (`grid.best_params_`), so I can see what the ensemble adds. Right now I only print the ensemble's scores.
- Save the fitted encoder and trained ensemble to disk (for example with `joblib`) so I can reuse them in a web app.
- Fix the `fuel_type` fill in `main.py`. It uses `.mode()` (a Series) instead of `.mode()[0]`, so most missing fuel types end up as `missing` instead of the most common fuel type.
