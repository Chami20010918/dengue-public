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
    .system-name { text-align: center; font-size: 4rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #00FFFF, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION ---
DISTRICTS = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612, "file": "FINAL_DASHBOARD_colombo.csv", "threshold": 2000},
    "Katugastota": {"lat": 7.3256, "lon": 80.6211, "file": "FINAL_DASHBOARD_katugastota.csv", "threshold": 300},
    "Ratnapura": {"lat": 6.6828, "lon": 80.3990, "file": "FINAL_DASHBOARD_ratnapura.csv", "threshold": 400}
}

# --- 4. DATA LOADING & GLOBAL STATUS ---
# We load ALL data first to build the map
map_data = []
current_district_data = None
global_max_risk = "SAFE"

for name, info in DISTRICTS.items():
    try:
        df = pd.read_csv(info["file"])
        last_pred = df.iloc[-1]['predicted_cases' if 'predicted_cases' in df.columns else 'predicted']
        
        # Determine Color & Status
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
            "name": name,
            "lat": info["lat"],
            "lon": info["lon"],
            "cases": int(last_pred),
            "color": color,
            "status": status
        })
    except:
        continue # Skip if file missing

# --- 5. HEADER ---
st.markdown("<div class='ministry-header'>MINISTRY OF HEALTH - SRI LANKA</div>", unsafe_allow_html=True)
st.markdown("<div class='board-header'>NATIONAL DENGUE CONTROL UNIT</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🦟 AUTODENGUE.LK</h1>", unsafe_allow_html=True)

# --- 6. NATIONAL MAP (ALL DISTRICTS) ---
st.divider()
c1, c2 = st.columns([3, 1])

with c1:
    st.subheader("🗺️ National Live Risk Map")
    st.caption("🔴 Red = Action Needed | 🟠 Orange = Alert | 🟢 Green = Stable")
    
    # Create the Map Layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(map_data),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=5000,
        pickable=True,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        line_color=[255, 255, 255]
    )
    
    # Tooltip for Map
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

# --- 7. DRILL DOWN SECTION ---
st.divider()
st.subheader("🔍 Detailed District Analysis")
selected = st.selectbox("Select Region to Analyze", list(DISTRICTS.keys()))

# Load selected data for graph
config = DISTRICTS[selected]
df_sel = pd.read_csv(config["file"])
df_sel['date'] = pd.to_datetime(df_sel['date'])
col_name = 'predicted_cases' if 'predicted_cases' in df_sel.columns else 'predicted'

st.line_chart(df_sel.set_index('date')[col_name], color="#00FFFF")

# --- 8. ACTIONABLE GUIDELINES (FRIENDLY & SIMPLE) ---
st.divider()
st.subheader("📋 Operational Guidelines (Simple Steps)")

tab1, tab2 = st.tabs(["👮 For Public Health Officers (PHIs)", "🏡 For General Public"])

with tab1:
    st.info("💡 **PHI Instructions: Focus on Prevention, not Panic.**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. Target High-Density Areas**
        * Check your area map. Go to the 3 most crowded streets.
        * Focus on construction sites and schools first.
        """)
        
    with c2:
        st.markdown("""
        **2. Constructive Engagement**
        * Do not issue fines immediately. Give a 3-day warning to clean.
        * Help residents identify "invisible" breeding spots (gutters, fridge trays).
        """)

with tab2:
    st.success("💡 **Public Instructions: Small Steps Save Lives.**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. The '10-Minute' Sunday Routine**
        * Spend just 10 minutes every Sunday looking for water.
        * Check: Flower pots, yoghurt cups, and roof gutters.
        """)
        
    with c2:
        st.markdown("""
        **2. Personal Protection**
        * Use mosquito repellent if going out in the morning/evening.
        * Keep windows closed during dawn (6-8 AM) and dusk (5-7 PM).
        """)

# --- 9. AI SIMULATOR ---
st.divider()
st.subheader("🤖 Weather Simulator")
rain = st.slider("Rainfall (mm)", 0, 500, 100)
st.metric("AI Predicted Impact", f"+ {int(rain * 0.5)} Extra Cases likely")
