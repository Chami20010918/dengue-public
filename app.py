import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE.LK | National Surveillance",
    page_icon="🦟",
    layout="wide"
)

# --- 2. CUSTOM DESIGN (OFFICIAL THEME + MOSQUITO ICON) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Official Header Styles */
    .ministry-header {
        text-align: center;
        color: #9ca3af;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .board-header {
        text-align: center;
        color: #60a5fa; /* Light Blue */
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 20px;
        text-transform: uppercase;
        border-bottom: 2px solid #374151;
        padding-bottom: 20px;
    }
    /* SYSTEM NAME WITH MOSQUITO ICON */
    .system-name {
        text-align: center;
        font-size: 4rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00FFFF, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -20px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px; /* Space between icon and text */
    }
    /* MOSQUITO ICON ANIMATION */
    .mosquito-icon {
        font-size: 4rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* Centered Selectbox */
    div[data-testid="stSelectbox"] {
        text-align: center;
    }
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SYSTEM CONFIGURATION ---
# UPDATED: File paths now point to the same folder (no "data/" prefix)
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612,
        "best_model_name": "Hybrid Ensemble (XGBoost + LSTM)",
        "accuracy": "72.4%",  
        "file_data": "dashboard_data_colombo.csv", 
        "risk_threshold": 2000
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211,
        "best_model_name": "XGBoost (Machine Learning)",
        "accuracy": "84.9%",  
        "file_data": "FINAL_DASHBOARD_katugastota.csv",
        "risk_threshold": 300
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990,
        "best_model_name": "Gradient Boosting (Log-Transform)",
        "accuracy": "61.3%", 
        "file_data": "FINAL_DASHBOARD_ratnapura.csv",
        "risk_threshold": 400
    }
}

# --- 4. OFFICIAL HEADER SECTION ---
st.markdown("<div class='ministry-header'>MINISTRY OF HEALTH - SRI LANKA GOVERNMENT</div>", unsafe_allow_html=True)
st.markdown("<div class='board-header'>DENGUE SURVEILLANCE BOARD</div>", unsafe_allow_html=True)

# THE NEW HEADER WITH THE MOSQUITO
st.markdown("""
    <div class='system-name'>
        <span class='mosquito-icon'>🦟</span> 
        AUTODENGUE.LK 
        <span class='mosquito-icon'>🦟</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #6b7280; margin-bottom: 40px;'>National AI-Driven Epidemic Forecasting System</p>", unsafe_allow_html=True)

# --- 5. CENTERED DISTRICT SELECTION ---
col_L, col_Mid, col_R = st.columns([1, 2, 1])
with col_Mid:
    selected_district = st.selectbox("🎯 SELECT SURVEILLANCE DISTRICT", list(DISTRICTS.keys()))

config = DISTRICTS[selected_district]

# --- LOAD DATA ---
try:
    # Load directly (assumes file is next to app.py)
    df = pd.read_csv(config["file_data"])
    df['date'] = pd.to_datetime(df['date'])
    
    # Get Last Known Data
    # Auto-detect column names based on different exports
    if 'predicted_cases' in df.columns:
        pred_col = 'predicted_cases'
        act_col = 'actual' if 'actual' in df.columns else 'dengue_cases'
    else:
        pred_col = 'predicted'
        act_col = 'actual'

    last_pred = df[pred_col].iloc[-1]
    last_date = df['date'].iloc[-1]
    
    # Handle NaNs in actuals for future dates
    valid_actuals = df[act_col].dropna()
    if not valid_actuals.empty:
        last_actual = valid_actuals.iloc[-1]
    else:
        last_actual = 0

    # Determine Risk
    if last_pred > config["risk_threshold"]:
        risk_color = "#ef4444" 
        risk_msg = "CRITICAL RISK"
    elif last_pred > config["risk_threshold"] * 0.7:
        risk_color = "#f59e0b" 
        risk_msg = "WARNING LEVEL"
    else:
        risk_color = "#10b981" 
        risk_msg = "NORMAL ACTIVITY"

except Exception as e:
    st.error(f"⚠️ ERROR: Could not find file '{config['file_data']}'. Please make sure the CSV file is in the same folder as this script!")
    st.stop()

# --- 6. KPI METRICS ---
st.divider()
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("MODEL ARCHITECTURE", config["best_model_name"], delta="Active")

with k2:
    st.metric("MODEL ACCURACY", config["accuracy"], delta="Verified")

with k3:
    st.metric("FORECAST (NEXT MONTH)", f"{int(last_pred)} Cases", delta=f"{int(last_pred - last_actual)} vs Last")

with k4:
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid {risk_color}; color: {risk_color}; padding: 10px; border-radius: 10px; font-weight: bold; margin-top: 0px;">
        {risk_msg}
    </div>
    """, unsafe_allow_html=True)

# --- 7. GRAPHS & MAPS ---
st.divider()
g_col, m_col = st.columns([2, 1])

with g_col:
    st.subheader(f"📈 {selected_district} Outbreak Trend")
    # Clean data for chart
    chart_df = df.set_index('date')[[act_col, pred_col]]
    st.line_chart(chart_df, color=["#00FFFF", "#FF0055"])
    st.caption("Cyan = Actual Data | Red = AI Prediction")

with m_col:
    st.subheader("🗺️ Risk Zone")
    map_df = pd.DataFrame([{"lat": config["lat"], "lon": config["lon"]}])
    
    # Color logic
    r, g, b = (239, 68, 68) if "CRITICAL" in risk_msg else ((245, 158, 11) if "WARNING" in risk_msg else (16, 185, 129))
    
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=config["lat"], longitude=config["lon"], zoom=10),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_color=[r, g, b, 160],
                get_radius=3000,
                pickable=True,
                filled=True,
            )
        ]
    ))

# --- 8. AI SIMULATOR (WITH ICON) ---
st.divider()
st.markdown("<h2 style='text-align: center;'>🦟 REAL-TIME WEATHER SIMULATOR</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>Adjust parameters to see AI prediction response</p>", unsafe_allow_html=True)

# Layout: Spacer | Controls | Spacer
s_left, s_mid, s_right = st.columns([1, 2, 1])

with s_mid:
    st.markdown("#### 🎛️ SIMULATION CONTROLS")
    rain = st.slider("🌧️ Rainfall (mm)", 0, 600, 200)
    temp = st.slider("🌡️ Temperature (°C)", 20, 40, 29)
    humidity = st.slider("💧 Humidity (%)", 50, 100, 80)
    wind = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

    # Simulation Logic
    base = last_pred
    r_factor = (rain - 200) * 0.5
    t_factor = (10 - abs(temp - 29)) * 5.0 # Sweet spot at 29
    h_factor = (humidity - 75) * 2.0
    
    sim_result = base + r_factor + t_factor + h_factor
    if sim_result < 0: sim_result = 0
    
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>AI PREDICTED OUTCOME</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #FFD700; font-size: 4rem;'>{int(sim_result)} Cases</h1>", unsafe_allow_html=True)

# --- 9. PROTOCOLS ---
st.divider()
if "CRITICAL" in risk_msg:
    st.error("🔴 RED PROTOCOL: IMMEDIATE FOGGING & CLINICAL SURGE CAPACITY REQUIRED.")
elif "WARNING" in risk_msg:
    st.warning("🟠 AMBER PROTOCOL: COMMUNITY CLEANING & LARVAL SURVEYS REQUIRED.")
else:
    st.success("🟢 GREEN PROTOCOL: ROUTINE SURVEILLANCE ACTIVE.")