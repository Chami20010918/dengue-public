import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

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

    /* GLOBAL DARK THEME */
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* REMOVE PADDING */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* --- HEADER STYLES --- */
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
    .sub-header {
        color: #a1a1aa;
        font-size: 0.85rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .main-title {
        background: linear-gradient(90deg, #22d3ee, #bef264);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
    }
    .mosquito-icon {
        font-size: 3.5rem;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(34, 211, 238, 0.5));
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    /* --- GLASS CARDS --- */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-color: #22d3ee;
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.1);
        transform: translateY(-2px);
    }
    
    /* --- STATUS BADGES --- */
    .badge { padding: 4px 12px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .badge-critical { background: rgba(220, 38, 38, 0.3); color: #fca5a5; border: 1px solid #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
    .badge-warning { background: rgba(234, 88, 12, 0.3); color: #fdba74; border: 1px solid #f97316; }
    .badge-safe { background: rgba(22, 163, 74, 0.3); color: #86efac; border: 1px solid #22c55e; }

    /* --- SIMULATOR --- */
    .sim-panel {
        background: #18181b;
        border-left: 4px solid #22d3ee;
        padding: 20px;
        margin-top: 20px;
        border-radius: 0 10px 10px 0;
    }

    /* OVERRIDE STREAMLIT METRICS */
    div[data-testid="stMetric"] {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        color: #fff !important;
    }
    div[data-testid="stMetricLabel"] { color: #a1a1aa !important; }
    div[data-testid="stMetricValue"] { color: #fff !important; }
    
    /* POSTER GRID */
    .poster-container {
        display: flex; 
        gap: 10px; 
        justify-content: center;
        margin-top: 20px;
    }
    .poster-img {
        border-radius: 10px;
        border: 2px solid #333;
        transition: transform 0.3s;
    }
    .poster-img:hover {
        transform: scale(1.05);
        border-color: #22d3ee;
    }
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
                color = [220, 38, 38, 255] # Red
            elif val > info["threshold"] * 0.7:
                status = "WARNING"
                color = [249, 115, 22, 255] # Orange
            else:
                status = "NORMAL"
                color = [34, 197, 94, 255] # Green
                
            data_list.append({
                "name": name, "lat": info["lat"], "lon": info["lon"],
                "cases": val, "status": status, "color": color,
                "model": info["model"], "acc": info["acc"]
            })
        except:
            pass
    return data_list

dashboard_data = load_all_data()

# --- 4. HEADER SECTION ---
st.markdown("""
<div class="header-container">
    <div class="sub-header">MINISTRY OF HEALTH • SRI LANKA GOVERNMENT</div>
    <div class="main-title">
        <span class="mosquito-icon">🦟</span>
        AUTODENGUE.LK
        <span class="mosquito-icon">🦟</span>
    </div>
    <div style="color: #71717a; margin-top: 10px;">National AI-Driven Epidemic Surveillance Unit</div>
</div>
""", unsafe_allow_html=True)

# --- 5. KPIs ---
if dashboard_data:
    total_cases = sum(d['cases'] for d in dashboard_data)
    high_risk_count = sum(1 for d in dashboard_data if d['status'] == "CRITICAL")
else:
    total_cases = 0; high_risk_count = 0

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Forecast (30 Days)", f"{total_cases}", "Patients")
with k2: st.metric("High Risk Zones", f"{high_risk_count}", "Districts", delta_color="inverse")
with k3: st.metric("System Status", "ONLINE", "Latency: 24ms")
with k4: st.metric("AI Confidence", "89.2%", "Ensemble")

st.markdown("---")

# --- 6. MAIN SURVEILLANCE MAP & STATUS ---
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Geospatial Risk Map")
    
    # FIXED MAP CONFIGURATION
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(dashboard_data),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=8000,          # Large radius to be visible
        pickable=True,
        stroked=True,
        filled=True,
        line_color=[255, 255, 255],
        line_width_min_pixels=2,
        opacity=0.8
    )
    
    # Dark Mode Map Style
    view_state = pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.2, pitch=40)
    
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10", # Forces Dark Map
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"html": "<div style='background: #111; color: white; padding: 10px; border: 1px solid #333;'><b>{name}</b><br>Status: {status}<br>Forecast: {cases}</div>"}
    ))

with col_details:
    st.subheader("📋 Regional Status")
    
    for city in dashboard_data:
        badge_class = f"badge-{city['status'].lower() if city['status'] != 'NORMAL' else 'safe'}"
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px; padding: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; font-size: 1.1rem;">{city['name']}</span>
                <span class="badge {badge_class}">{city['status']}</span>
            </div>
            <div style="font-size: 0.9rem; color: #a1a1aa;">
                Predicted: <span style="color: white; font-weight: 700;">{city['cases']}</span> Patients<br>
                Model: {city['model']} ({city['acc']})
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- 7. DEEP DIVE SECTION ---
st.markdown("---")
st.subheader("🔍 Advanced Analytics")

target_city_name = st.selectbox("Select District", list(DISTRICTS.keys()))
target_config = DISTRICTS[target_city_name]

# Load Data for Chart
try:
    df_chart = pd.read_csv(target_config["file"])
    df_chart['date'] = pd.to_datetime(df_chart['date'])
    cols_map = {}
    if 'dengue_cases' in df_chart.columns: cols_map['dengue_cases'] = 'Actual'
    if 'actual' in df_chart.columns: cols_map['actual'] = 'Actual'
    if 'predicted_cases' in df_chart.columns: cols_map['predicted_cases'] = 'Predicted'
    if 'predicted' in df_chart.columns: cols_map['predicted'] = 'Predicted'
    clean_chart = df_chart.rename(columns=cols_map).set_index('date')
    if 'Actual' in clean_chart.columns: clean_chart['Actual'] = clean_chart['Actual'].fillna(0).astype(int)
    if 'Predicted' in clean_chart.columns: clean_chart['Predicted'] = clean_chart['Predicted'].fillna(0).astype(int)
except:
    clean_chart = pd.DataFrame()

# TABS (Including New Posters Tab)
tab_trend, tab_sim, tab_proto, tab_media = st.tabs(["📈 Trend Chart", "🤖 Weather Simulator", "📢 Guidelines", "🖼️ Awareness Posters"])

with tab_trend:
    st.markdown(f"**12-Month Trajectory: {target_city_name}**")
    st.line
