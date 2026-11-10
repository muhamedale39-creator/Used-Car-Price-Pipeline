from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Used Car Price Prediction API",
    description="ML API powered by Random Forest & XGBoost Voting Ensemble",
    version="1.0.0"
)

MODEL_PATH = "car_artifacts.joblib"
artifacts = None

@app.on_event("startup")
def load_artifacts():
    global artifacts
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Artifact {MODEL_PATH} not found!")

class CarData(BaseModel):
    brand: str
    model: str
    milage: float
    fuel_type: str
    transmission: str
    ext_col: str
    int_col: str
    accident: int = Field(..., description="1 if accident reported, 0 otherwise")
    clean_title: int = Field(..., description="1 if clean title, 0 otherwise")
    hp: float
    liters: float
    vehicle_age: int
    mileage_per_year: float

@app.get("/")
def home():
    return {"message": "Welcome to Used Car Price API. Visit /docs to test."}

@app.post("/predict")
def predict_price(data: CarData):
    global artifacts
    if not artifacts:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        
        transformer = artifacts["transformer"]
        model = artifacts["model"]
        
        processed_input = transformer.transform(input_df)
        prediction = model.predict(processed_input)
        
        predicted_price = float(prediction[0])
        
        return {
            "predicted_price_usd": round(predicted_price, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))