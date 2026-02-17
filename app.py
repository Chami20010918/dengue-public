import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE | Command Center",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. DARK CYBERPUNK CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

.stApp {
    background-color: #000000;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}

.block-container { padding-top: 1rem; padding-bottom: 2rem; }

.header-container {
    border-bottom: 1px solid #333;
    padding-bottom: 20px;
    margin-bottom: 30px;
    text-align: center;
    background: #09090b;
    padding-top: 20px;
    border-radius: 15px;
    border: 1px solid #27272a;
}

.main-title {
    background: linear-gradient(90deg, #22d3ee, #bef264);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 900;
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s;
}

.badge { padding: 4px 12px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
.badge-critical { background: rgba(220, 38, 38, 0.3); color: #fca5a5; border: 1px solid #ef4444; }
.badge-warning { background: rgba(234, 88, 12, 0.3); color: #fdba74; border: 1px solid #f97316; }
.badge-safe { background: rgba(22, 163, 74, 0.3); color: #86efac; border: 1px solid #22c55e; }

div[data-testid="stMetric"] {
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# --- DISTRICT CONFIG ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612,
        "file": "FINAL_DASHBOARD_colombo.csv",
        "threshold": 2000, "model": "Hybrid Ensemble", "acc": "72.4%"
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211,
        "file": "FINAL_DASHBOARD_katugastota.csv",
        "threshold": 300, "model": "XGBoost ML", "acc": "84.9%"
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990,
        "file": "FINAL_DASHBOARD_ratnapura.csv",
        "threshold": 400, "model": "Gradient Boost", "acc": "61.3%"
    }
}

# --- LOAD DATA ---
@st.cache_data
def load_all_data():
    data_list = []
    latest_month_name = ""

    for name, info in DISTRICTS.items():
        try:
            df = pd.read_csv(info["file"])
            df['date'] = pd.to_datetime(df['date'])

            # 🔥 Remove 2020-2022
            df = df[df['date'].dt.year >= 2023]

            pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'

            latest_row = df.sort_values("date").iloc[-1]
            val = int(round(latest_row[pred_col]))

            latest_month_name = latest_row['date'].strftime("%B %Y")

            if val > info["threshold"]:
                status = "CRITICAL"
                color = [220, 38, 38, 255]
            elif val > info["threshold"] * 0.7:
                status = "WARNING"
                color = [249, 115, 22, 255]
            else:
                status = "NORMAL"
                color = [34, 197, 94, 255]

            data_list.append({
                "name": name,
                "lat": info["lat"],
                "lon": info["lon"],
                "cases": val,
                "status": status,
                "color": color,
                "model": info["model"],
                "acc": info["acc"]
            })

        except:
            pass

    return data_list, latest_month_name


dashboard_data, latest_month = load_all_data()

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div class="main-title">🦟 AUTODENGUE.LK</div>
    <div style="color: #71717a;">National AI-Driven Epidemic Surveillance Unit</div>
</div>
""", unsafe_allow_html=True)

# --- KPI SECTION WITH ANIMATION ---
if dashboard_data:
    total_cases = sum(d['cases'] for d in dashboard_data)
    high_risk = sum(1 for d in dashboard_data if d['status'] == "CRITICAL")
else:
    total_cases = 0
    high_risk = 0

k1, k2, k3, k4 = st.columns(4)

# 🔥 Animated Counter
counter_placeholder = k1.empty()
for i in range(0, total_cases+1, max(1, total_cases//50)):
    counter_placeholder.metric(f"{latest_month} Forecast", f"{i}")
    time.sleep(0.01)

k2.metric("High Risk Zones", high_risk)
k3.metric("System Status", "ONLINE")
k4.metric("AI Confidence", "89.2%")

st.markdown("---")

# --- MAP ---
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Geospatial Risk Map")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(dashboard_data),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=8000,
        pickable=True
    )

    view_state = pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.2)

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer]
    ))

with col_details:
    st.subheader("📋 Regional Status")
    for city in dashboard_data:
        badge_class = f"badge-{city['status'].lower() if city['status'] != 'NORMAL' else 'safe'}"
        st.markdown(f"""
        <div class="metric-card">
            <b>{city['name']}</b>
            <span class="badge {badge_class}">{city['status']}</span><br>
            Predicted: <b>{city['cases']}</b><br>
            Model: {city['model']} ({city['acc']})
        </div>
        """, unsafe_allow_html=True)

# --- ADVANCED ANALYTICS ---
st.markdown("---")
st.subheader("🔍 Advanced Analytics")

target = st.selectbox("Select District", list(DISTRICTS.keys()))
target_config = DISTRICTS[target]

try:
    df_chart = pd.read_csv(target_config["file"])
    df_chart['date'] = pd.to_datetime(df_chart['date'])

    # 🔥 Remove 2020-2022
    df_chart = df_chart[df_chart['date'].dt.year >= 2023]

    pred_col = 'predicted_cases' if 'predicted_cases' in df_chart.columns else 'predicted'
    actual_col = 'dengue_cases' if 'dengue_cases' in df_chart.columns else 'actual'

    clean_chart = df_chart.rename(columns={
        actual_col: "Actual",
        pred_col: "Predicted"
    }).set_index("date")

except:
    clean_chart = pd.DataFrame()

tab1, tab2 = st.tabs(["📈 Trend Chart", "📢 Guidelines"])

with tab1:
    st.line_chart(clean_chart[['Actual', 'Predicted']])

with tab2:
    st.success("🏡 Public Advisory")
    st.markdown("• Weekly 10-min clean-up")
    st.markdown("• Remove standing water")
    st.markdown("• Use mosquito repellent")
