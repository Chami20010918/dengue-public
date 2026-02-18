import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE | Command Center",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED CYBERPUNK CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }

    .header-container {
        border-bottom: 1px solid #333;
        padding: 25px;
        margin-bottom: 30px;
        text-align: center;
        background: #09090b;
        border-radius: 15px;
        border: 1px solid #27272a;
    }
    
    .main-title-container { display: flex; justify-content: center; align-items: center; gap: 20px; }
    .main-title {
        background: linear-gradient(90deg, #22d3ee, #bef264);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8rem; font-weight: 900; margin: 0;
    }

    .mosquito-logo {
        font-size: 3.5rem;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(34, 211, 238, 0.5));
    }

    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); border-color: rgba(239, 68, 68, 1); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); border-color: rgba(239, 68, 68, 0.5); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); border-color: rgba(239, 68, 68, 1); }
    }

    .badge { padding: 4px 12px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .badge-critical { 
        background: #450a0a !important; color: #fca5a5 !important; 
        border: 2px solid #ef4444 !important; animation: pulse-red 2s infinite; 
    }
    .badge-warning { background: rgba(234, 88, 12, 0.2); color: #fdba74; border: 1px solid #f97316; }
    .badge-safe { background: rgba(22, 163, 74, 0.2); color: #86efac; border: 1px solid #22c55e; }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; transition: 0.3s;
    }
    .critical-border { border-left: 5px solid #ef4444 !important; background: rgba(239, 68, 68, 0.05); }

    .emergency-banner {
        background: #7f1d1d; color: white; padding: 15px; border-radius: 10px;
        text-align: center; font-weight: 800; margin-bottom: 20px; border: 2px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION & DATA ---
DISTRICTS = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612, "file": "FINAL_DASHBOARD_colombo.csv", "threshold": 2000, "model": "Hybrid Ensemble", "acc": "72.4%"},
    "Katugastota": {"lat": 7.3256, "lon": 80.6211, "file": "FINAL_DASHBOARD_katugastota.csv", "threshold": 300, "model": "XGBoost ML", "acc": "84.9%"},
    "Ratnapura": {"lat": 6.6828, "lon": 80.3990, "file": "FINAL_DASHBOARD_ratnapura.csv", "threshold": 400, "model": "Gradient Boost", "acc": "61.3%"}
}

@st.cache_data
def load_all_data():
    data_list = []
    target_date = pd.to_datetime("2026-02-01") 
    for name, info in DISTRICTS.items():
        try:
            df = pd.read_csv(info["file"])
            df['date'] = pd.to_datetime(df['date'])
            current_row = df[df['date'] == target_date]
            if not current_row.empty:
                pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'
                val = int(round(current_row.iloc[0][pred_col]))
                radius = 18000 if val > info["threshold"] else 8000
                if val > info["threshold"]: status, color = "CRITICAL", [239, 68, 68, 180]
                elif val > info["threshold"] * 0.7: status, color = "WARNING", [249, 115, 22, 180]
                else: status, color = "NORMAL", [34, 197, 94, 180]
                data_list.append({"name": name, "lat": info["lat"], "lon": info["lon"], "cases": val, "status": status, "color": color, "model": info["model"], "acc": info["acc"], "radius": radius})
        except: pass
    return data_list

dashboard_data = load_all_data()

# Notification Logic
critical_zones = [d['name'] for d in dashboard_data if d['status'] == "CRITICAL"]
if critical_zones: st.toast(f"🚨 OUTBREAK ALERT: {len(critical_zones)} Critical Zones", icon="🚨")

# --- 4. HEADER ---
st.markdown(f"""
<div class="header-container">
    <div class="sub-header">MINISTRY OF HEALTH • SRI LANKA GOVERNMENT</div>
    <div class="main-title-container">
        <span class="mosquito-logo">🦟</span>
        <h1 class="main-title">AUTODENGUE.LK</h1>
    </div>
</div>
""", unsafe_allow_html=True)

if critical_zones:
    st.markdown(f'<div class="emergency-banner">ALERT: Outbreak Levels in {", ".join(critical_zones)} for Feb 2026</div>', unsafe_allow_html=True)

# --- 5. KPIs ---
total_cases = sum(d['cases'] for d in dashboard_data)
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Feb '26 Forecast", f"{total_cases}", "Total Cases")
with k2: st.metric("Critical Zones", f"{len(critical_zones)}", delta_color="inverse")
with k3: st.metric("System Health", "STABLE")
with k4: st.metric("Avg. Accuracy", "82.1%")

st.markdown("---")

# --- 6. MAP & DETAILS ---
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Risk Map (February 2026)")
    view_state = pdk.ViewState(latitude=7.8731, longitude=80.7718, zoom=6.5, pitch=40)
    layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame(dashboard_data), get_position="[lon, lat]", get_color="color", get_radius="radius", pickable=True, opacity=0.4, filled=True, stroked=True, line_width_min_pixels=2, line_color=[255, 255, 255, 150])
    # Change map_style to a simple string and set map_provider to 'carto'
st.pydeck_chart(pdk.Deck(
    map_provider="carto",  # Use Carto instead of Mapbox
    map_style="dark",      # Options: 'light' or 'dark'
    initial_view_state=view_state, 
    layers=[layer],
    tooltip={"text": "{name}\nForecast: {cases}\nStatus: {status}"}
))
with col_details:
    st.subheader("📋 Regional Status")
    for city in dashboard_data:
        is_crit = "critical-border" if city['status'] == "CRITICAL" else ""
        badge_type = f"badge-{city['status'].lower() if city['status'] != 'NORMAL' else 'safe'}"
        st.markdown(f'<div class="metric-card {is_crit}" style="margin-bottom:12px;"><div style="display:flex; justify-content:space-between;"><span style="font-weight:800;">{city["name"]}</span><span class="badge {badge_type}">{city["status"]}</span></div><div style="margin-top:10px; font-size:0.9rem;">Forecast: <b style="color:white;">{city["cases"]} cases</b><br>Confidence: <span style="color:#22d3ee;">{city["acc"]}</span></div></div>', unsafe_allow_html=True)

# --- 7. TRENDS & SIMULATOR ---
st.markdown("---")
target_city = st.selectbox("Detailed Analysis", list(DISTRICTS.keys()))

# Robust Variable Initialization
clean_chart = pd.DataFrame()

try:
    df_chart = pd.read_csv(DISTRICTS[target_city]["file"])
    df_chart['date'] = pd.to_datetime(df_chart['date'])
    df_chart = df_chart[df_chart['date'] >= '2023-01-01']
    df_chart = df_chart.rename(columns={'predicted_cases':'Predicted', 'predicted':'Predicted', 'actual':'Actual', 'dengue_cases':'Actual'})
    plot_cols = [c for c in ['Actual', 'Predicted'] if c in df_chart.columns]
    clean_chart = df_chart.set_index('date')[plot_cols].fillna(0)
except:
    st.error("Data source error for this district.")

t_trend, t_sim = st.tabs(["📈 Trend Chart", "🤖 Weather Simulator"])

with t_trend:
    if not clean_chart.empty:
        st.line_chart(clean_chart, color=["#22d3ee", "#ef4444"])
    else:
        st.warning("No trend data available.")

with t_sim:
    st.markdown(f"### ⚡ Scenario Dengue Case Test: {target_city}")
    base_val = next((d['cases'] for d in dashboard_data if d['name'] == target_city), 0)
    
    # 4-Column Layout for Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1: rain = st.slider("Rainfall (mm)", 0, 600, 200)
    with c2: temp = st.slider("Temp (°C)", 22, 38, 29)
    with c3: hum = st.slider("Humidity (%)", 50, 100, 75)
    with c4: wind = st.slider("Wind (km/h)", 0, 60, 15)
    
    # SIMULATION LOGIC
    # High wind speeds (> 20 km/h) reduce breeding efficiency
    sim_delta = int(((rain - 200) * 0.5) + ((temp - 29) * 12) + ((hum - 75) * 4) - ((wind - 15) * 3))
    sim_final = max(0, base_val + sim_delta)
    
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        st.metric("Simulated Forecast", f"{sim_final}", delta=f"{sim_delta}")
    with s_col2: 
        if wind > 40:
            st.info("🍃 Strong winds detected: Breeding efficiency reduced despite other factors.")
        elif sim_delta > 50:
            st.warning("🚨 High Risk: Environment favoring rapid mosquito proliferation.")
        else:
            st.success("Stable: No significant environmental threat detected.")


