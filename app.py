import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

API_URL = "https://used-car-price-pipeline.onrender.com"

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_cars_data.csv")
    return df

df = load_data()

st.title("🚗 Used Car Price Prediction System")
st.markdown("Estimate the market price of a used car based on its specifications, mileage, and features.")

st.sidebar.header("Car Specifications")

def user_input_features():
    brands = sorted(df['brand'].dropna().unique().tolist())
    brand = st.sidebar.selectbox("Brand", options=brands)
    
    filtered_models = sorted(df[df['brand'] == brand]['model'].dropna().unique().tolist())
    model = st.sidebar.selectbox("Model", options=filtered_models)
    
    milage = st.sidebar.number_input("Mileage (miles)", min_value=0, max_value=300000, value=45000)
    
    fuel_types = sorted(df['fuel_type'].dropna().unique().tolist())
    fuel_type = st.sidebar.selectbox("Fuel Type", options=fuel_types)
    
    transmissions = sorted(df['transmission'].dropna().unique().tolist())
    transmission = st.sidebar.selectbox("Transmission", options=transmissions)
    
    ext_cols = sorted(df['ext_col'].dropna().unique().tolist())
    ext_col = st.sidebar.selectbox("Exterior Color", options=ext_cols)
    
    int_cols = sorted(df['int_col'].dropna().unique().tolist())
    int_col = st.sidebar.selectbox("Interior Color", options=int_cols)
    
    accident = st.sidebar.selectbox("Accident Reported?", options=[("None reported", 0), ("At least 1 accident", 1)], format_func=lambda x: x[0])[1]
    clean_title = st.sidebar.selectbox("Clean Title?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    
    hp = st.sidebar.number_input("Horsepower (HP)", min_value=50.0, max_value=1000.0, value=325.0)
    liters = st.sidebar.number_input("Engine Size (Liters)", min_value=0.5, max_value=8.0, value=3.5)
    vehicle_age = st.sidebar.number_input("Vehicle Age (Years)", min_value=0, max_value=50, value=4)
    mileage_per_year = milage / (vehicle_age + 1)

    return {
        "brand": str(brand).lower().strip(),
        "model": str(model).lower().strip(),
        "milage": float(milage),
        "fuel_type": str(fuel_type).lower().strip(),
        "transmission": str(transmission).lower().strip(),
        "ext_col": str(ext_col).lower().strip(),
        "int_col": str(int_col).lower().strip(),
        "accident": int(accident),
        "clean_title": int(clean_title),
        "hp": float(hp),
        "liters": float(liters),
        "vehicle_age": int(vehicle_age),
        "mileage_per_year": float(mileage_per_year)
    }

input_data = user_input_features()

st.subheader("Selected Vehicle Summary")
st.json(input_data)

if st.button("Predict Market Price"):
    with st.spinner("Connecting to backend and calculating price..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()
                price = result.get("predicted_price_usd")
                st.success("Prediction generated successfully!")
                st.metric(label="Estimated Market Price", value=f"${price:,.2f}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")