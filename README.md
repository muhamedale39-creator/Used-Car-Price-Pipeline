# 🚗 Used-Car Price Prediction Pipeline & API

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end machine learning system that takes a messy, raw used-car dataset from Kaggle, cleans and engineers features using Pandas, trains a tuned Voting Ensemble (Random Forest + XGBoost), exposes predictions via an asynchronous **FastAPI** backend containerized with **Docker**, and serves users through an interactive **Streamlit** web application featuring dynamic category dropdowns to eliminate input errors.

---

## 🚀 Live Demo & Endpoints

- **Interactive Streamlit App:** [View Live App](https://used-car-price-pipeline-5x7ggj7gsmod2i8w8rwm5z.streamlit.app/)
- **FastAPI Backend Documentation:** [API Swagger Docs](https://used-car-price-pipeline.onrender.com/docs)

---

## 📊 Dataset
The automotive data used to build this pipeline is sourced directly from the [Kaggle Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).

---

## 🏗️ Project Architecture & Workflow

```
data.csv ──► main.py ──► cleaned_cars_data.csv ──► train.py ──► car_artifacts.joblib ──► FastAPI (main.py) ──► Streamlit (app.py)
 (raw)     cleaning, features,      (model-ready)     target encoding, 80/20 split,        Saved Model Pipeline        Render Cloud          Streamlit Cloud
           outlier capping, EDA                        GridSearchCV (RF + XGBoost),                                  
                │                                      VotingRegressor                                             
                └──► images/ (EDA charts)
```

- `main.py` cleans the raw data, engineers new features, caps outliers, and generates EDA charts.
- `train.py` encodes text columns via target encoding, tunes models with cross-validation grid search, builds a voting ensemble, and saves artifacts to `car_artifacts.joblib`.
- `main.py` (FastAPI) runs the production backend API loading the saved model artifacts (`/predict` endpoint).
- `app.py` runs the interactive Streamlit frontend UI with automated dataset-driven dropdown selectors.

---

## 🧹 Data Cleaning & Feature Engineering (`main.py`)

1. **Numeric parsing:** Stripped symbols and commas from `price` and `milage` using regex and converted them to numbers.
2. **Regex feature extraction from `engine`:** Extracted `hp` (horsepower) and `liters` (engine size) from free text descriptions.
3. **Three-tier missing value imputation:** Filled missing values for `hp` and `liters` using a fallback chain (median of brand & model $\rightarrow$ median of brand $\rightarrow$ global median).
4. **Categorical cleanup:** Standardized `fuel_type`, `accident`, and `clean_title` binary flags.
5. **Brand and model name fixes:** Repaired truncated brand names (`Land` → `Land Rover`, `Aston` → `Aston Martin`, `Alfa` → `Alfa Romeo`) and stripped prefix residues.
6. **New features:** Created `vehicle_age` (`2026 − model_year`) and `mileage_per_year`, dropping `model_year`.
7. **Text normalization:** Lowercased, trimmed, and handled missing text fields.
8. **Outlier capping:** Capped extreme `price` outliers using the IQR rule.
9. **Exploratory charts:** Saved visual plots (`price_distribution.png`, `mileage_vs_price.png`, `top_10_brands.png`) into `images/`.

---

## 🤖 Modeling & Training (`train.py`)

1. **Split:** 80% train / 20% test split.
2. **Target Encoding:** Encoded categorical features (`brand`, `model`, `fuel_type`, `transmission`, `ext_col`, `int_col`) using `TargetEncoder` inside a `ColumnTransformer` fitted exclusively on the training set to prevent data leakage.
3. **Random Forest & XGBoost Tuning:** Tuned individual regressors using 5-fold cross-validation and GridSearchCV optimizing for $R^2$ score.
4. **Voting Ensemble:** Combined the optimized Random Forest and XGBoost estimators into a `VotingRegressor`.

### Evaluation Metrics (Ensemble Performance)

| Metric | Score |
| --- | --- |
| **$R^2$ score** | 0.9083 (90.8% of data variance explained) |
| **MAE** | \$5,586.37 |
| **RMSE** | \$8,605.27 |

---

## 📁 Project Structure

```text
Used-Car-Price-Pipeline/
├── Images/
│   ├── price_distribution.png
│   ├── mileage_vs_price.png
│   └── top_10_brands.png
├── main.py                # FastAPI Backend & /predict Endpoint
├── train.py               # Model Training & Artifact Generation
├── app.py                 # Streamlit Frontend with Dynamic Dropdowns
├── car_artifacts.joblib   # Saved Transformer & Voting Ensemble Pipeline
├── data.csv               # Raw Dataset
├── cleaned_cars_data.csv  # Cleaned Dataset & Dropdown Source
├── Dockerfile             # Container configuration for Render
├── requirements.txt       # Project dependencies
└── README.md
```

---

## ⚙️ How to Run It Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhamedale39-creator/Used-Car-Price-Pipeline.git
   cd Used-Car-Price-Pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Clean data & generate charts:**
   ```bash
   python main.py
   ```

4. **Train and save model artifacts:**
   ```bash
   python train.py
   ```

5. **Run the FastAPI Backend:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. **Run the Streamlit Frontend (in a separate terminal):**
   ```bash
   streamlit run app.py
   ```

---

## 🐳 Docker Deployment

To build and run the backend container locally:

```bash
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api
```

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, FastAPI, Streamlit, Docker, Git, GitHub