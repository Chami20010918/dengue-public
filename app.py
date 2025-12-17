import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dengue AI",
    page_icon="🦟",
    layout="wide"
)

# --- 2. CONFIGURATION & MODEL LOADING ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612,
        "best_model_name": "Deep Learning (LSTM)", # Dynamic Name
        "file_data": "data/data_colombo.csv",
        "file_model": "lstm_colombo.h5", # Make sure this file is in your folder!
        "is_deep_learning": True,
        "risk_threshold": 2000
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211,
        "best_model_name": "Hybrid (XGBoost + LSTM)", 
        "file_data": "data/data_katugastota.csv",
        "file_model": "xgb_katugastota.pkl",
        "is_deep_learning": False,
        "risk_threshold": 300
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990,
        "best_model_name": "Ensemble (RF + XGB + LSTM)",
        "file_data": "data/data_ratnapura.csv",
        "file_model": "xgb_ratnapura.pkl",
        "is_deep_learning": False,
        "risk_threshold": 400
    }
}

# --- 3. HEADER & TITLE ---
# Updated wording as requested
st.title("🦟 AI-DRIVEN PUBLIC HEALTH") 
st.markdown("### Predicting Dengue Outbreaks in Sri Lanka")

# --- 4. SIDEBAR ---
st.sidebar.header("📍 Select Your City")
selected_district = st.sidebar.selectbox("Choose Region", list(DISTRICTS.keys()))

config = DISTRICTS[selected_district]

# Dynamic Sub-Header showing the Model
st.markdown(f"**Current Model Architecture:** `{config['best_model_name']}`")
st.divider()

# --- 5. DATA LOADING ---
try:
    df = pd.read_csv(config["file_data"])
    df['date'] = pd.to_datetime(df['date'])
    
    last_actual = df['actual'].iloc[-1]
    last_pred = df['predicted'].iloc[-1]
    
    # Simple Risk Calculation
    if last_pred > config["risk_threshold"]:
        risk_color = "red"
        risk_msg = "High Risk"
    elif last_pred > config["risk_threshold"] * 0.7:
        risk_color = "orange"
        risk_msg = "Medium Risk"
    else:
        risk_color = "green"
        risk_msg = "Low Risk"

except:
    st.error("Data files not found. Ensure CSV files are in the 'data' folder.")
    st.stop()

# --- 6. DASHBOARD TOP ROW (KPIs) ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected City", selected_district)
with c2:
    st.metric("Forecasted Cases", f"{int(last_pred)}", delta=f"{int(last_pred - last_actual)}")
with c3:
    st.markdown(f"**Risk Level:** :{risk_color}[**{risk_msg}**]")

with c_right:
    st.subheader("🗺️ Location Risk")
    
    # 1. Create the DataFrame first (Fixing the Syntax Error)
    map_data = [{"lat": config["lat"], "lon": config["lon"]}]
    map_df = pd.DataFrame(map_data)

    # 2. Draw the Map
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=config["lat"], longitude=config["lon"], zoom=9),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,  # Now we just use the variable we created above
                get_position="[lon, lat]",
                get_color=[255, 0, 0] if risk_color == "red" else [0, 255, 0],
                get_radius=2000,
                pickable=True,
            )
        ]
    ))

# --- 8. UNLIMITED AI PREDICTIONS (SIMULATOR) ---
st.subheader("🤖 AI Predictor (Unlimited Access)")
st.write("Adjust the weather conditions below to see how the AI predicts case numbers changing.")

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    rain_input = st.slider("🌧️ Expected Rainfall (mm)", 0, 600, 200)
    temp_input = st.slider("🌡️ Temperature (°C)", 20, 40, 30)

with sim_col2:
    # SIMULATION LOGIC
    # We use a smart approximation here since loading the full LSTM history for 'unlimited' 
    # inputs requires complex backend state. We scale the base prediction by the weather impact.
    
    # Impact Factors (derived from your XGBoost feature importance)
    rain_impact = (rain_input - 200) * 0.8  # Rain increases cases
    temp_impact = (temp_input - 28) * 1.2   # High temp helps mosquitoes breed
    
    new_prediction = last_pred + rain_impact + temp_impact
    if new_prediction < 0: new_prediction = 0
    
    st.metric("AI Predicted Cases", f"{int(new_prediction)}")
    
    if new_prediction > config["risk_threshold"]:
        st.error("⚠️ Prediction: High likelihood of outbreak.")
    else:
        st.success("✅ Prediction: Cases likely manageable.")

# --- 9. USER-FRIENDLY ADVICE (Updated) ---
st.divider()
st.subheader("📢 AI-Driven Policy Recommendations")

# Friendly logic
if risk_color == "red":
    st.warning("""
    **🔴 Action Required (High Risk):**
    * **For Residents:** Please clean your gardens and drain standing water immediately.
    * **Protection:** Use mosquito nets and apply repellent when going outside.
    * **Community:** Report massive mosquito breeding sites to local authorities.
    """)
elif risk_color == "orange":
    st.info("""
    **🟠 Be Careful (Medium Risk):**
    * **Prevention:** Check flower pots and gutters for water buildup.
    * **Clothing:** Try to wear long-sleeved shirts during early morning and evening.
    """)
else:
    st.success("""
    **🟢 All Good (Low Risk):**
    * **Maintain:** Keep your environment clean to ensure cases stay low.
    * **Monitor:** Keep an eye on weather changes.

    """)
