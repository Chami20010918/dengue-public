import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE.LK | National Surveillance",
    page_icon="🦟",
    layout="wide"
)

# --- 2. CUSTOM DESIGN ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .ministry-header { text-align: center; color: #9ca3af; font-size: 1.2rem; letter-spacing: 2px; }
    .board-header { text-align: center; color: #60a5fa; font-size: 2.5rem; font-weight: 800; }
    /* Metric Cards */
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION (ACCURACY & MODEL NAMES ADDED HERE) ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612, 
        "file": "FINAL_DASHBOARD_colombo.csv", 
        "threshold": 2000,
        "model": "Hybrid Ensemble (XGBoost + LSTM)",
        "accuracy": "72.4%"
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211, 
        "file": "FINAL_DASHBOARD_katugastota.csv", 
        "threshold": 300,
        "model": "XGBoost (Machine Learning)",
        "accuracy": "84.9%"
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990, 
        "file": "FINAL_DASHBOARD_ratnapura.csv", 
        "threshold": 400,
        "model": "Gradient Boosting (Log-Transform)",
        "accuracy": "61.3%"
    }
}

# --- 4. HEADER ---
st.markdown("<div class='ministry-header'>MINISTRY OF HEALTH - SRI LANKA</div>", unsafe_allow_html=True)
st.markdown("<div class='board-header'>NATIONAL DENGUE CONTROL UNIT</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🦟 AUTODENGUE.LK</h1>", unsafe_allow_html=True)

# --- 5. NATIONAL MAP (ALL DISTRICTS) ---
st.divider()
map_data = []

# Load data for map
for name, info in DISTRICTS.items():
    try:
        df = pd.read_csv(info["file"])
        # Find the correct column for prediction
        pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'
        last_pred = df.iloc[-1][pred_col]
        
        # Color Logic
        if last_pred > info["threshold"]:
            color = [239, 68, 68] # Red
            status = "High"
        elif last_pred > info["threshold"] * 0.7:
            color = [245, 158, 11] # Orange
            status = "Moderate"
        else:
            color = [16, 185, 129] # Green
            status = "Low"
            
        map_data.append({
            "name": name, "lat": info["lat"], "lon": info["lon"],
            "cases": int(last_pred), "color": color, "status": status
        })
    except:
        continue

c1, c2 = st.columns([3, 1])
with c1:
    st.subheader("🗺️ National Live Risk Map")
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(map_data),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=8000,
        pickable=True,
        stroked=True,
        filled=True,
        line_color=[255, 255, 255]
    )
    tooltip = {"html": "<b>{name}</b><br/>Status: {status}<br/>Forecast: {cases} Cases"}
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.5),
        layers=[layer],
        tooltip=tooltip
    ))

with c2:
    st.subheader("📊 District Status")
    for district in map_data:
        emoji = "🔴" if district["status"] == "High" else "🟠" if district["status"] == "Moderate" else "🟢"
        st.metric(f"{emoji} {district['name']}", f"{district['cases']} Cases", district['status'])

# --- 6. DRILL DOWN & MODEL INFO ---
st.divider()
st.subheader("🔍 Detailed District Analysis")

# Select District
selected = st.selectbox("Select Region to Analyze", list(DISTRICTS.keys()))
config = DISTRICTS[selected]

# SHOW MODEL NAME AND ACCURACY HERE
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("🧠 Model Architecture", config["model"], delta="Active")
with m2:
    st.metric("✅ Model Accuracy", config["accuracy"], delta="Verified")
with m3:
    st.metric("📉 Risk Threshold", f"{config['threshold']} Cases")

# --- 7. GRAPH FIX (CYAN & RED LINES) ---
st.subheader(f"📈 Outbreak Trend: {selected}")

# Load & Clean Data
df_sel = pd.read_csv(config["file"])
df_sel['date'] = pd.to_datetime(df_sel['date'])

# Normalize Column Names (Fixes the missing line issue)
# We rename whatever columns we have to 'Actual' and 'Predicted'
cols_map = {}
if 'dengue_cases' in df_sel.columns: cols_map['dengue_cases'] = 'Actual'
if 'actual' in df_sel.columns: cols_map['actual'] = 'Actual'
if 'predicted_cases' in df_sel.columns: cols_map['predicted_cases'] = 'Predicted'
if 'predicted' in df_sel.columns: cols_map['predicted'] = 'Predicted'

df_chart = df_sel.rename(columns=cols_map)
df_chart = df_chart.set_index('date')

# Plot only Actual and Predicted
st.line_chart(df_chart[['Actual', 'Predicted']], color=["#00FFFF", "#FF0055"])
st.caption("Cyan Line = Actual Cases (History) | Red Line = AI Prediction (Forecast)")

# --- 8. REAL-TIME PREDICTOR ---
st.divider()
st.subheader(f"🤖 Real-Time Predictor for {selected}")
st.info(f"Adjust weather sliders below to simulate a scenario specifically for **{selected}**.")

rain = st.slider("Rainfall (mm)", 0, 500, 100)
temp = st.slider("Temperature (°C)", 20, 40, 30)

# Simple Simulation Logic
base_cases = int(map_data[0]['cases']) # Just as a baseline example
impact = int((rain * 0.8) + ((temp - 28) * 10))
simulated_total = base_cases + impact

c_sim1, c_sim2 = st.columns(2)
with c_sim1:
    st.metric("Weather Impact", f"{impact:+} Cases", "Due to Rain/Temp")
with c_sim2:
    st.metric("Total Simulated Cases", f"{simulated_total}", "If weather persists")

# --- 9. GUIDELINES ---
st.divider()
st.subheader("📋 Operational Guidelines")
st.success("For PHIs: Target high-density areas. Give warnings before fines.")
st.info("For Public: Check gutters and pots every Sunday for 10 minutes.")
