import streamlit as st
import requests

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

API_URL = "https://car-backend-url.onrender.com/predict"

st.title("🚗 Used Car Price Prediction System")
st.markdown("Estimate the market price of a used car based on its specifications, mileage, and features.")

st.sidebar.header("Car Specifications")

def user_input_features():
    brand = st.sidebar.text_input("Brand", value="ford")
    model = st.sidebar.text_input("Model", value="f-150 xlt")
    milage = st.sidebar.number_input("Mileage (miles)", min_value=0, max_value=300000, value=45000)
    fuel_type = st.sidebar.selectbox("Fuel Type", options=["gasoline", "hybrid", "diesel", "e85 flex fuel", "plug-in hybrid", "electric"])
    transmission = st.sidebar.text_input("Transmission", value="10-speed automatic")
    ext_col = st.sidebar.text_input("Exterior Color", value="black")
    int_col = st.sidebar.text_input("Interior Color", value="black")
    accident = st.sidebar.selectbox("Accident Reported?", options=[("None reported", 0), ("At least 1 accident", 1)], format_func=lambda x: x[0])[1]
    clean_title = st.sidebar.selectbox("Clean Title?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    hp = st.sidebar.number_input("Horsepower (HP)", min_value=50.0, max_value=1000.0, value=325.0)
    liters = st.sidebar.number_input("Engine Size (Liters)", min_value=0.5, max_value=8.0, value=3.5)
    vehicle_age = st.sidebar.number_input("Vehicle Age (Years)", min_value=0, max_value=50, value=4)
    mileage_per_year = milage / (vehicle_age + 1)

    return {
        "brand": brand.lower().strip(),
        "model": model.lower().strip(),
        "milage": float(milage),
        "fuel_type": fuel_type,
        "transmission": transmission.lower().strip(),
        "ext_col": ext_col.lower().strip(),
        "int_col": int_col.lower().strip(),
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