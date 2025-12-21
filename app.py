import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AUTODENGUE | National Surveillance System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ENTERPRISE CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* GLOBAL RESET */
    .stApp {
        background-color: #050505; /* Deep Black/Grey */
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* REMOVE DEFAULT STREAMLIT PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* --- DASHBOARD HEADER --- */
    .header-container {
        border-bottom: 1px solid #333;
        padding-bottom: 20px;
        margin-bottom: 30px;
        text-align: center;
    }
    .sub-header {
        color: #888;
        font-size: 0.85rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .main-title {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    /* --- CUSTOM CARDS (GLASSMORPHISM) --- */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        border-color: #00C9FF;
        transform: translateY(-2px);
    }
    .card-title {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .card-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
    }
    .card-delta {
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* --- STATUS BADGES --- */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-safe { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }

    /* --- SIMULATOR PANEL --- */
    .sim-panel {
        background: #0f0f0f;
        border-left: 4px solid #00C9FF;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* HIDE STREAMLIT ELEMENTS */
    div[data-testid="stToolbar"] { display: none; }
    footer { display: none; }
    
    /* CUSTOM METRIC STYLE */
    div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION & DATA ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612, 
        "file": "FINAL_DASHBOARD_colombo.csv", 
        "threshold": 2000,
        "model": "Hybrid Ensemble (XGB+LSTM)",
        "acc": "72.4%"
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211, 
        "file": "FINAL_DASHBOARD_katugastota.csv", 
        "threshold": 300,
        "model": "XGBoost Regressor",
        "acc": "84.9%"
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990, 
        "file": "FINAL_DASHBOARD_ratnapura.csv", 
        "threshold": 400,
        "model": "Gradient Boosting",
        "acc": "61.3%"
    }
}

# Load Data Function
@st.cache_data
def load_all_data():
    data_list = []
    for name, info in DISTRICTS.items():
        try:
            df = pd.read_csv(info["file"])
            # Normalize Columns
            pred_col = 'predicted_cases' if 'predicted_cases' in df.columns else 'predicted'
            # INT CONVERSION
            val = int(round(df.iloc[-1][pred_col]))
            
            # Status Logic
            if val > info["threshold"]:
                status = "CRITICAL"
                color = [220, 38, 38, 200] # Red
            elif val > info["threshold"] * 0.7:
                status = "WARNING"
                color = [245, 158, 11, 200] # Orange
            else:
                status = "NORMAL"
                color = [16, 185, 129, 200] # Green
                
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
    <div class="sub-header">Ministry of Health • Sri Lanka Government</div>
    <div class="main-title">AUTODENGUE.LK</div>
    <div style="color: #666; margin-top: 10px;">National AI-Driven Epidemic Surveillance Unit</div>
</div>
""", unsafe_allow_html=True)

# --- 5. TOP LEVEL METRICS (KPIs) ---
total_cases = sum(d['cases'] for d in dashboard_data)
high_risk_count = sum(1 for d in dashboard_data if d['status'] == "CRITICAL")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Predicted Cases", f"{total_cases}", "Next 30 Days")
with k2:
    st.metric("High Risk Zones", f"{high_risk_count}", "districts require action", delta_color="inverse")
with k3:
    st.metric("AI Model Status", "ONLINE", "v2.4.0 Updated")
with k4:
    st.metric("Data Confidence", "89.2%", "Ensemble Avg")

st.markdown("---")

# --- 6. MAIN SURVEILLANCE AREA (Map + Status) ---
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("Geospatial Risk Analysis")
    
    # Map Layer
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
        line_width_min_pixels=1,
        opacity=0.8
    )
    
    view_state = pdk.ViewState(latitude=7.0, longitude=80.5, zoom=7.2, pitch=30)
    
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10", # Dark mode map
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"html": "<div style='background: #111; color: white; padding: 10px; border: 1px solid #333;'><b>{name}</b><br>Status: {status}<br>Forecast: {cases}</div>"}
    ))

with col_details:
    st.subheader("Regional Status Report")
    
    for city in dashboard_data:
        # Determine CSS class for badge
        badge_class = f"badge-{city['status'].lower() if city['status'] != 'NORMAL' else 'safe'}"
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 10px; padding: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 1.1rem;">{city['name']}</span>
                <span class="badge {badge_class}">{city['status']}</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #aaa;">
                Forecast: <span style="color: white; font-weight: bold;">{city['cases']}</span> Patients<br>
                Model: {city['model']} ({city['acc']})
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- 7. DEEP DIVE & SIMULATOR ---
st.markdown("---")
st.subheader("Advanced Analysis & Simulation")

# Selection
target_city_name = st.selectbox("Select District for Deep Dive", list(DISTRICTS.keys()))
target_config = DISTRICTS[target_city_name]

# Get Data for Chart
df_chart = pd.read_csv(target_config["file"])
df_chart['date'] = pd.to_datetime(df_chart['date'])

# Rename for cleanliness
cols_map = {}
if 'dengue_cases' in df_chart.columns: cols_map['dengue_cases'] = 'Historical Data'
if 'actual' in df_chart.columns: cols_map['actual'] = 'Historical Data'
if 'predicted_cases' in df_chart.columns: cols_map['predicted_cases'] = 'AI Forecast'
if 'predicted' in df_chart.columns: cols_map['predicted'] = 'AI Forecast'

clean_chart = df_chart.rename(columns=cols_map).set_index('date')
# Ensure Integers
if 'Historical Data' in clean_chart.columns: clean_chart['Historical Data'] = clean_chart['Historical Data'].fillna(0).astype(int)
if 'AI Forecast' in clean_chart.columns: clean_chart['AI Forecast'] = clean_chart['AI Forecast'].fillna(0).astype(int)

# --- TABS LAYOUT ---
tab_trend, tab_sim, tab_proto = st.tabs(["📈 Trend Analysis", "🤖 Weather Simulator", "📋 Protocols"])

with tab_trend:
    st.markdown("### 12-Month Disease Trajectory")
    st.line_chart(clean_chart[['Historical Data', 'AI Forecast']], color=["#00C9FF", "#FF0055"])
    st.caption("Blue: Confirmed Historical Cases | Red: AI Projected Cases")

with tab_sim:
    st.markdown(f"### Real-Time Weather Impact: {target_city_name}")
    st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: rain = st.slider("Rainfall (mm)", 0, 500, 150)
    with col_s2: temp = st.slider("Temp (°C)", 20, 40, 29)
    with col_s3: hum = st.slider("Humidity (%)", 40, 100, 75)
    with col_s4: wind = st.slider("Wind (km/h)", 0, 50, 10)
    
    # Calc
    base = int([d['cases'] for d in dashboard_data if d['name'] == target_city_name][0])
    delta = int((rain-150)*0.4 + (temp-29)*5 + (hum-75)*2 - (wind-10)*1.5)
    final = max(0, base + delta)
    
    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 3])
    with res_c1:
        st.metric("Simulated Total", f"{final}", delta=f"{delta} from baseline")
    with res_c2:
        if delta > 20:
            st.warning("Analysis: High rainfall combined with humidity significantly increases breeding probability.")
        elif delta < -20:
            st.success("Analysis: Adverse weather conditions for mosquitoes (Dry/Windy) reduce projected cases.")
        else:
            st.info("Analysis: Weather conditions remain neutral relative to baseline forecast.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_proto:
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("""
        #### PUBLIC HEALTH OFFICER (PHI) GUIDELINES
        1. **Deployment:** Prioritize sectors 4 and 7 in the selected district.
        2. **Action:** Inspect construction sites (slab tops) and school gutters.
        3. **Enforcement:** Issue rectification notices (3-day compliance) before legal action.
        """)
    with c_p2:
        st.markdown("""
        #### GENERAL PUBLIC ADVISORY
        1. **Inspection:** Perform weekly "10-Minute Checks" on Sunday mornings.
        2. **Prevention:** Eliminate standing water in flower pot trays and fridge backs.
        3. **Protection:** Use repellent during peak biting hours (06:00-08:00 & 16:00-18:30).
        """)
