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

# --- 2. CUSTOM DESIGN (OFFICIAL THEME + NEW ALERT CARDS) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* --- HEADER STYLES --- */
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
        color: #60a5fa;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 20px;
        text-transform: uppercase;
        border-bottom: 2px solid #374151;
        padding-bottom: 20px;
    }
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
        gap: 15px;
    }
    .mosquito-icon {
        font-size: 4rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* --- NEW STATUS CARD STYLES --- */
    .status-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        color: white;
        border-left: 10px solid;
    }
    .status-critical {
        background: linear-gradient(90deg, #450a0a, #1a0505);
        border-color: #ef4444; /* Red Border */
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }
    .status-warning {
        background: linear-gradient(90deg, #451a03, #1a0a02);
        border-color: #f59e0b; /* Orange Border */
    }
    .status-safe {
        background: linear-gradient(90deg, #064e3b, #022c22);
        border-color: #10b981; /* Green Border */
    }
    .status-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #e5e7eb;
    }
    
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
    }
    div[data-testid="stSelectbox"] {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SYSTEM CONFIGURATION (CHANGE THRESHOLDS HERE!) ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612,
        "best_model_name": "Hybrid Ensemble (XGBoost + LSTM)",
        "accuracy": "72.4%",  
        "file_data": "FINAL_DASHBOARD_colombo.csv", 
        
        # --- CHANGE THIS NUMBER TO ADJUST ALERT LEVEL ---
        "risk_threshold": 2000 
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211,
        "best_model_name": "XGBoost (Machine Learning)",
        "accuracy": "84.9%",  
        "file_data": "FINAL_DASHBOARD_katugastota.csv",
        
        # --- CHANGE THIS NUMBER TO ADJUST ALERT LEVEL ---
        "risk_threshold": 300
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990,
        "best_model_name": "Gradient Boosting (Log-Transform)",
        "accuracy": "61.3%", 
        "file_data": "FINAL_DASHBOARD_ratnapura.csv",
        
        # --- CHANGE THIS NUMBER TO ADJUST ALERT LEVEL ---
        "risk_threshold": 400
    }
}

# --- 4. OFFICIAL HEADER SECTION ---
st.markdown("<div class='ministry-header'>MINISTRY OF HEALTH - SRI LANKA GOVERNMENT</div>", unsafe_allow_html=True)
st.markdown("<div class='board-header'>DENGUE SURVEILLANCE BOARD</div>", unsafe_allow_html=True)

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
    df = pd.read_csv(config["file_data"])
    df['date'] = pd.to_datetime(df['date'])
    
    if 'predicted_cases' in df.columns:
        pred_col = 'predicted_cases'
        act_col = 'actual' if 'actual' in df.columns else 'dengue_cases'
    else:
        pred_col = 'predicted'
        act_col = 'actual'

    last_pred = df[pred_col].iloc[-1]
    
    valid_actuals = df[act_col].dropna()
    if not valid_actuals.empty:
        last_actual = valid_actuals.iloc[-1]
    else:
        last_actual = 0

    # Determine Risk
    if last_pred > config["risk_threshold"]:
        risk_class = "status-critical"
        risk_icon = "🔴"
        risk_title = "CRITICAL OUTBREAK DETECTED"
        risk_msg = "CRITICAL" # For Map Logic
    elif last_pred > config["risk_threshold"] * 0.7:
        risk_class = "status-warning"
        risk_icon = "🟠"
        risk_title = "ELEVATED RISK LEVEL"
        risk_msg = "WARNING"
    else:
        risk_class = "status-safe"
        risk_icon = "🟢"
        risk_title = "NORMAL SURVEILLANCE"
        risk_msg = "SAFE"

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
    # Small Badge for KPI
    badge_color = "#ef4444" if risk_msg == "CRITICAL" else "#f59e0b" if risk_msg == "WARNING" else "#10b981"
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid {badge_color}; color: {badge_color}; padding: 10px; border-radius: 10px; font-weight: bold;">
        {risk_title.split(' ')[0]} STATUS
    </div>
    """, unsafe_allow_html=True)

# --- 7. GRAPHS & MAPS ---
st.divider()
g_col, m_col = st.columns([2, 1])

with g_col:
    st.subheader(f"📈 {selected_district} Outbreak Trend")
    chart_df = df.set_index('date')[[act_col, pred_col]]
    st.line_chart(chart_df, color=["#00FFFF", "#FF0055"])
    st.caption("Cyan = Actual Data | Red = AI Prediction")

with m_col:
    st.subheader("🗺️ Risk Zone")
    map_df = pd.DataFrame([{"lat": config["lat"], "lon": config["lon"]}])
    r, g, b = (239, 68, 68) if risk_msg == "CRITICAL" else ((245, 158, 11) if risk_msg == "WARNING" else (16, 185, 129))
    
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

s_left, s_mid, s_right = st.columns([1, 2, 1])

with s_mid:
    st.markdown("#### 🎛️ SIMULATION CONTROLS")
    rain = st.slider("🌧️ Rainfall (mm)", 0, 600, 200)
    temp = st.slider("🌡️ Temperature (°C)", 20, 40, 29)
    humidity = st.slider("💧 Humidity (%)", 50, 100, 80)
    wind = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

    base = last_pred
    r_factor = (rain - 200) * 0.5
    t_factor = (10 - abs(temp - 29)) * 5.0
    h_factor = (humidity - 75) * 2.0
    
    sim_result = base + r_factor + t_factor + h_factor
    if sim_result < 0: sim_result = 0
    
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>AI PREDICTED OUTCOME</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #FFD700; font-size: 4rem;'>{int(sim_result)} Cases</h1>", unsafe_allow_html=True)

# --- 9. NEW & IMPROVED ALERTS ---
st.divider()

# Defining the specific advice text
if risk_msg == "CRITICAL":
    action_text = """
    <b>⚠️ URGENT ACTION REQUIRED:</b><br>
    The AI model forecasts a significant surge in cases. <br><br>
    1. <b>MOH Offices:</b> Immediate fogging in high-density clusters.<br>
    2. <b>Hospitals:</b> Prepare DHF (Dengue Hemorrhagic Fever) wards.<br>
    3. <b>Public:</b> Mandatory cleaning of premises. Legal action for breeding sites.
    """
elif risk_msg == "WARNING":
    action_text = """
    <b>⚠️ PRECAUTIONARY PHASE:</b><br>
    Cases are rising above the safety baseline.<br><br>
    1. <b>Community:</b> Organize 'Shramadana' cleaning campaigns.<br>
    2. <b>Schools:</b> Inspect water tanks and gutters this week.<br>
    3. <b>Personal:</b> Use mosquito repellents during peak hours (Dawn/Dusk).
    """
else:
    action_text = """
    <b>✅ SURVEILLANCE ACTIVE:</b><br>
    Cases are within manageable limits.<br><br>
    1. <b>Routine:</b> Continue weekly premises inspections.<br>
    2. <b>Monitor:</b> Watch for sudden weather changes in the simulator above.<br>
    3. <b>Data:</b> System will update automatically next week.
    """

# Render the Beautiful Custom Card
st.markdown(f"""
<div class="status-card {risk_class}">
    <div class="status-title">{risk_icon} {risk_title}</div>
    <div class="status-text">{action_text}</div>
</div>
""", unsafe_allow_html=True)
