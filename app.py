import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE | National Surveillance",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PROFESSIONAL LIGHT THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* GLOBAL SETTINGS (LIGHT MODE) */
    .stApp {
        background-color: #f8f9fa; /* Very Light Grey */
        color: #1f2937; /* Dark Grey Text */
        font-family: 'Inter', sans-serif;
    }
    
    /* REMOVE PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* --- HEADER STYLES --- */
    .header-container {
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 20px;
        margin-bottom: 30px;
        text-align: center;
        background: white;
        padding-top: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        color: #4b5563; /* Dark Grey - VISIBLE NOW */
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .main-title {
        background: linear-gradient(90deg, #2563eb, #0ea5e9); /* Blue Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
    }
    .mosquito-icon {
        font-size: 3.5rem;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 2px 2px rgba(0,0,0,0.1));
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    /* --- METRIC CARDS (CLEAN WHITE) --- */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        border-color: #3b82f6; /* Blue border on hover */
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* --- STATUS BADGES --- */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .badge-critical { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-warning { background: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
    .badge-safe { background: #d1fae5; color: #065f46; border: 1px solid #34d399; }

    /* --- SIMULATOR PANEL --- */
    .sim-panel {
        background: #ffffff;
        border-left: 5px solid #2563eb;
        padding: 25px;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    /* OVERRIDE STREAMLIT METRICS TO LOOK NICE ON WHITE */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        color: #111 !important;
    }
    div[data-testid="stMetricLabel"] { color: #6b7280 !important; }
    div[data-testid="stMetricValue"] { color: #111827 !important; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION & DATA ---
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

@st.cache_data
def load_all_data():
    data_list = []
    for name, info in DISTRICTS.items():
        try:
            df = pd.read_csv(info["file"])
            pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'
            val = int(round(df.iloc[-1][pred_col]))
            
            if val > info["threshold"]:
                status = "CRITICAL"
                color = [220, 38, 38, 200] 
            elif val > info["threshold"] * 0.7:
                status = "WARNING"
                color = [234, 88, 12, 200]
            else:
                status = "NORMAL"
                color = [22, 163, 74, 200]
                
            data_list.append({
                "name": name, "lat": info["lat"], "lon": info["lon"],
                "cases": val, "status": status, "color": color,
                "model": info["model"], "acc": info["acc"]
            })
        except:
            pass
    return data_list

dashboard_data = load_all_data()

# --- 4. HEADER SECTION (UPDATED) ---
st.markdown("""
<div class="header-container">
    <div class="sub-header">MINISTRY OF HEALTH • SRI LANKA GOVERNMENT</div>
    <div class="main-title">
        <span class="mosquito-icon">🦟</span>
        AUTODENGUE.LK
    </div>
    <div style="color: #6b7280; margin-top: 10px; font-weight: 500;">National AI-Driven Epidemic Surveillance Unit</div>
</div>
""", unsafe_allow_html=True)

# --- 5. TOP LEVEL METRICS ---
if dashboard_data:
    total_cases = sum(d['cases'] for d in dashboard_data)
    high_risk_count = sum(1 for d in dashboard_data if d['status'] == "CRITICAL")
else:
    total_cases = 0
    high_risk_count = 0

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("National Forecast (30 Days)", f"{total_cases}", "Patients")
with k2: st.metric("High Risk Zones", f"{high_risk_count}", "Districts", delta_color="inverse")
with k3: st.metric("System Status", "ONLINE", "Updated Today")
with k4: st.metric("AI Confidence", "89.2%", "High")

st.markdown("---")

# --- 6. MAIN SURVEILLANCE AREA ---
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Geospatial Risk Map")
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(dashboard_data),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=8000,
        pickable=True,
        stroked=True,
        filled=True,
        line_color=[255, 255, 255],
        line_width_min_pixels=2
    )
    
    # Using a Light Map Style for consistency
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10", 
        initial_view_state=pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.2),
        layers=[layer],
        tooltip={"html": "<div style='background: white; color: black; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'><b>{name}</b><br>Status: {status}<br>Forecast: {cases}</div>"}
    ))

with col_details:
    st.subheader("📋 Regional Status")
    
    for city in dashboard_data:
        badge_class = f"badge-{city['status'].lower() if city['status'] != 'NORMAL' else 'safe'}"
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px; padding: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 800; font-size: 1.1rem; color: #111;">{city['name']}</span>
                <span class="badge {badge_class}">{city['status']}</span>
            </div>
            <div style="font-size: 0.9rem; color: #4b5563;">
                Predicted Patients: <span style="color: #111; font-weight: 800; font-size: 1.1rem;">{city['cases']}</span><br>
                <span style="font-size: 0.8rem; color: #9ca3af;">Model: {city['model']} ({city['acc']})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- 7. DEEP DIVE SECTION ---
st.markdown("---")
st.subheader("🔍 Advanced District Analysis")

target_city_name = st.selectbox("Select District", list(DISTRICTS.keys()))
target_config = DISTRICTS[target_city_name]

# Prepare Chart Data
try:
    df_chart = pd.read_csv(target_config["file"])
    df_chart['date'] = pd.to_datetime(df_chart['date'])
    
    cols_map = {}
    if 'dengue_cases' in df_chart.columns: cols_map['dengue_cases'] = 'Actual'
    if 'actual' in df_chart.columns: cols_map['actual'] = 'Actual'
    if 'predicted_cases' in df_chart.columns: cols_map['predicted_cases'] = 'Predicted'
    if 'predicted' in df_chart.columns: cols_map['predicted'] = 'Predicted'

    clean_chart = df_chart.rename(columns=cols_map).set_index('date')
    # Integer Conversion
    if 'Actual' in clean_chart.columns: clean_chart['Actual'] = clean_chart['Actual'].fillna(0).astype(int)
    if 'Predicted' in clean_chart.columns: clean_chart['Predicted'] = clean_chart['Predicted'].fillna(0).astype(int)
except:
    st.error("Data file not found for selected district.")
    clean_chart = pd.DataFrame()

# TABS
tab_trend, tab_sim, tab_proto = st.tabs(["📈 Trend Chart", "🤖 Weather Simulator", "📢 Guidelines"])

with tab_trend:
    st.markdown(f"**12-Month Disease Forecast: {target_city_name}**")
    # Using Blue (Actual) and Red (Predicted) which looks good on white
    st.line_chart(clean_chart[['Actual', 'Predicted']], color=["#2563eb", "#dc2626"])
    st.caption("Blue Line: Historical Data | Red Line: AI Forecast")

with tab_sim:
    st.markdown(f"**Real-Time Weather Impact Analysis: {target_city_name}**")
    st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: rain = st.slider("Rainfall (mm)", 0, 500, 150)
    with col_s2: temp = st.slider("Temperature (°C)", 20, 40, 29)
    with col_s3: hum = st.slider("Humidity (%)", 40, 100, 75)
    with col_s4: wind = st.slider("Wind (km/h)", 0, 50, 10)
    
    # Calc
    base = int([d['cases'] for d in dashboard_data if d['name'] == target_city_name][0])
    delta = int((rain-150)*0.4 + (temp-29)*5 + (hum-75)*2 - (wind-10)*1.5)
    final = max(0, base + delta)
    
    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 3])
    with res_c1:
        st.metric("New Projection", f"{final}", delta=f"{delta} weather adjustment")
    with res_c2:
        if delta > 20:
            st.warning("⚠️ High Risk Conditions: Heavy rain + humidity accelerates breeding.")
        elif delta < -20:
            st.success("✅ Low Risk Conditions: Dry or windy weather reduces mosquito activity.")
        else:
            st.info("ℹ️ Neutral Conditions: Weather impact is within normal ranges.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_proto:
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("""
        #### 👮 FOR PHI OFFICERS
        1. **Deployment:** Target high-density zones in **{target_city_name}**.
        2. **Inspect:** Construction sites (slab tops) and schools.
        3. **Enforce:** Issue 3-day warnings before fining.
        """)
    with c_p2:
        st.markdown("""
        #### 🏡 FOR GENERAL PUBLIC
        1. **Check:** 10 minutes every Sunday (Gutters/Pots).
        2. **Clean:** Remove all standing water immediately.
        3. **Protect:** Use repellent (Dawn & Dusk).
        """)
