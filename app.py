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

# --- 2. ADVANCED UI STYLING (Glassmorphism) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Modern Cards */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Headers */
    .ministry-header { 
        text-align: center; color: #9ca3af; font-size: 1.1rem; letter-spacing: 3px; font-weight: 600; text-transform: uppercase;
    }
    .board-header { 
        text-align: center; color: #60a5fa; font-size: 2.2rem; font-weight: 800; margin-top: -10px; text-transform: uppercase;
    }
    
    /* Metric styling to remove decimals */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION ---
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
st.markdown("<div class='ministry-header'>MINISTRY OF HEALTH - SRI LANKA GOVERNMENT</div>", unsafe_allow_html=True)
st.markdown("<div class='board-header'>DENGUE SURVEILLANCE BOARD</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>🦟 AUTODENGUE.LK</h1>", unsafe_allow_html=True)

# --- 5. DATA LOADING & MAP PREP ---
map_data = []

for name, info in DISTRICTS.items():
    try:
        df = pd.read_csv(info["file"])
        pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'
        
        # FORCE INTEGER (No Decimals)
        last_pred = int(round(df.iloc[-1][pred_col]))
        
        # Status Logic
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
            "cases": last_pred, "color": color, "status": status
        })
    except:
        continue

# --- 6. TOP SECTION: MAP & STATUS ---
col_map, col_stat = st.columns([3, 1])

with col_map:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
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
        line_color=[255, 255, 255],
        line_width_min_pixels=2
    )
    
    # Tooltip matches the clean look
    tooltip = {"html": "<b>{name}</b><br/>Risk: {status}<br/>Forecast: {cases} Patients"}
    
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.5),
        layers=[layer],
        tooltip=tooltip
    ))
    st.markdown('</div>', unsafe_allow_html=True)

with col_stat:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📊 Live Status")
    for d in map_data:
        icon = "🔴" if d["status"] == "High" else "🟠" if d["status"] == "Moderate" else "🟢"
        st.metric(f"{icon} {d['name']}", f"{d['cases']}", "Patients")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. CITY DRILL DOWN ---
st.divider()
st.subheader("🔍 Regional Deep Dive")
selected = st.selectbox("Select District for Analysis", list(DISTRICTS.keys()))
config = DISTRICTS[selected]

# Model Info Cards
m1, m2, m3 = st.columns(3)
with m1: st.info(f"**Model:** {config['model']}")
with m2: st.success(f"**Accuracy:** {config['accuracy']}")
with m3: st.warning(f"**Alert Limit:** {config['threshold']} Patients")

# --- 8. THE CHART (WHOLE NUMBERS & BOTH LINES) ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.subheader(f"📈 12-Month Trend: {selected}")

# Load & Clean
df_sel = pd.read_csv(config["file"])
df_sel['date'] = pd.to_datetime(df_sel['date'])

# Rename cols for the Legend
cols_map = {}
if 'dengue_cases' in df_sel.columns: cols_map['dengue_cases'] = 'Actual'
if 'actual' in df_sel.columns: cols_map['actual'] = 'Actual'
if 'predicted_cases' in df_sel.columns: cols_map['predicted_cases'] = 'Predicted'
if 'predicted' in df_sel.columns: cols_map['predicted'] = 'Predicted'

df_chart = df_sel.rename(columns=cols_map).set_index('date')

# Ensure Integers
if 'Actual' in df_chart.columns: df_chart['Actual'] = df_chart['Actual'].fillna(0).astype(int)
if 'Predicted' in df_chart.columns: df_chart['Predicted'] = df_chart['Predicted'].fillna(0).astype(int)

# Plot
st.line_chart(df_chart[['Actual', 'Predicted']], color=["#00FFFF", "#FF0055"])
st.caption("Cyan = Actual History | Red = AI Prediction")
st.markdown('</div>', unsafe_allow_html=True)

# --- 9. REAL-TIME PREDICTOR (4 PARAMETERS) ---
st.divider()
st.markdown(f"<h2 style='text-align: center;'>🤖 AI Simulator: {selected}</h2>", unsafe_allow_html=True)
st.info("Adjust the 4 weather parameters below to see how the case count changes.")

st.markdown('<div class="css-card">', unsafe_allow_html=True)
c_sim1, c_sim2 = st.columns(2)

with c_sim1:
    rain = st.slider("🌧️ Rainfall (mm)", 0, 500, 150)
    temp = st.slider("🌡️ Temperature (°C)", 20, 40, 30)

with c_sim2:
    hum = st.slider("💧 Humidity (%)", 50, 100, 80)
    wind = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

# --- SIMULATION MATH (INTEGER LOGIC) ---
# 1. Get Baseline
base_cases = int(map_data[0]['cases']) # Baseline from current prediction

# 2. Calculate Coefficients (Simplified for UI)
rain_effect = (rain - 100) * 0.5    # More rain = more cases
temp_effect = (temp - 28) * 5       # Heat accelerates breeding
hum_effect = (hum - 75) * 2         # Humidity helps survival
wind_effect = (wind - 10) * -2      # Wind blows them away

total_change = int(rain_effect + temp_effect + hum_effect + wind_effect)
final_prediction = int(base_cases + total_change)

if final_prediction < 0: final_prediction = 0

# --- DISPLAY RESULT ---
st.markdown("---")
r1, r2, r3 = st.columns([1, 2, 1])
with r2:
    st.markdown(f"<h1 style='text-align: center; color: #FFD700; font-size: 5rem;'>{final_prediction}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Predicted Patients</p>", unsafe_allow_html=True)
    
    if total_change > 0:
        st.caption(f"⚠️ Weather conditions are adding approx. {total_change} extra cases.")
    else:
        st.caption(f"✅ Weather conditions are reducing approx. {abs(total_change)} cases.")
st.markdown('</div>', unsafe_allow_html=True)

# --- 10. GUIDELINES ---
st.divider()
st.subheader("📋 Official Protocols")
g1, g2 = st.columns(2)
with g1:
    st.error("**👮 For PHIs (Public Health Inspectors)**")
    st.markdown("- Focus on construction sites & schools.")
    st.markdown("- Issue warnings before fines (3-day grace period).")
with g2:
    st.success("**🏡 For General Public**")
    st.markdown("- **10-Minute Sunday:** Check gutters & pots.")
    st.markdown("- **Protection:** Use repellent at dawn & dusk.")
