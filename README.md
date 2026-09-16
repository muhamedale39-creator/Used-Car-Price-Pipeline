# Used-Car Price Prediction & Pipeline

This repository contains my complete Machine Learning project. I built a pipeline that takes a messy, raw dataset of used cars from Kaggle, cleans it, handles missing values using a smart looping strategy, and trains a Supervised Learning model to predict car prices.

Instead of just running a simple model in a messy Jupyter Notebook, I organized my code into clean, standalone Python scripts (`main.py` and `train.py`) to practice real-world coding standards.

##  What the Code Does (My Process)

### 1. Feature Extraction via Regex
The raw `engine` column was completely unorganized text (like `300.0HP 3.0L V6 Cylinder Engine`). I used Regular Expressions (Regex) to pull out the important numbers and created two fresh columns:
* `hp` (Horsepower)
* `liters` (Engine size)
Once the data was safely pulled out, I deleted the messy original text column.

### 2. 3-Tier Missing Value Imputation
Instead of just filling missing values with a basic global average (which would ruin the data accuracy), I used a **3-tier group fallback strategy** using Pandas `groupby`:
* **Tier 1:** Fill the missing value with the **median of that exact Car Brand and Model**.
* **Tier 2:** If that specific model data doesn't exist, fall back to the **overall Brand median**.
* **Tier 3:** If the brand is missing too, fall back to the **global dataset median**.

### 3. Smart Categorical For-Loop Encoding
To pass columns like `brand`, `model`, and `fuel_type` into an ML model, they have to be changed from text to numbers. I wrote a clean `for-loop` that automatically finds all text columns and uses `LabelEncoder` to change them to numeric badges together. 

I also saved each encoder inside a dictionary to make sure their memories aren't wiped, which keeps the project ready if I want to deploy it as a web app later.

### 4. Handling Memory Limitations (`MemoryError`)
While training my Random Forest model, my script originally crashed with a `MemoryError` because the trees were growing too deep on my laptop's RAM. I solved this by adding regularization and putting a ceiling on the trees (`max_depth=15`) and scaling back to 150 estimators. This stopped the crash completely and let the script run successfully.

---

##  Performance & Evaluation Scores

To check how smart my model is at predicting prices, I split the data into **80% Training** and **20% Testing**, and evaluated the model using multiple regression metrics:

* **R² Score:** My final accuracy grade showing how many pricing patterns my model successfully learned.
* **Mean Absolute Error (MAE):** The average dollar amount my guesses are off by.
* **Root Mean Squared Error (RMSE):** A score that penalizes large mistakes heavily, showing if the model makes any extreme guessing errors.
* **Max Error:** The single absolute worst-case mistake the model made on the test set.

### Project Folder View
* `main.py` - The script that handles feature extraction, cleaning, and outlier capping.
* `train.py` - The script that handles the encoding loop, splits the data, trains the Random Forest model, and prints the scores.
* `data.csv` - The raw Kaggle dataset.
* `cleaned_cars_data.csv` - The output file ready for machine learning.
* `requirements.txt` - The Python libraries needed to run this project.
* `images/` - Charts generated automatically during data exploration (EDA).

---

##  How to Run the Project Locally

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd Used-Car-Price-Pipeline
   ```

2. Install all required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the cleaning script to generate the clean CSV file:
   ```bash
   python main.py
   ```

4. Run the training script to encode the columns, build the model, and print the scores:
   ```bash
   python train.py
   ```
