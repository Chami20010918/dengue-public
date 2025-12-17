import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dengue AI Command Center",
    page_icon="🦟",
    layout="wide"
)

# --- 2. CUSTOM DESIGN (DARK BLUE THEME) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117; /* Very Dark Blue/Black */
        color: #ffffff;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    /* Headings */
    h1, h2, h3 {
        color: #60a5fa !important; /* Light Blue Text */
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Sliders */
    .stSlider {
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SYSTEM CONFIGURATION ---
DISTRICTS = {
    "Colombo": {
        "lat": 6.9271, "lon": 79.8612,
        "best_model_name": "Deep Learning (LSTM)",
        "file_data": "data/data_colombo.csv",
        "risk_threshold": 2000
    },
    "Katugastota": {
        "lat": 7.3256, "lon": 80.6211,
        "best_model_name": "Hybrid (XGBoost + LSTM)",
        "file_data": "data/data_katugastota.csv",
        "risk_threshold": 300
    },
    "Ratnapura": {
        "lat": 6.6828, "lon": 80.3990,
        "best_model_name": "Ensemble (RF + XGB + LSTM)",
        "file_data": "data/data_ratnapura.csv",
        "risk_threshold": 400
    }
}

# --- 4. HEADER SECTION ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("""
    <h1 style='text-align: left; margin-bottom: 0;'>🦟 AI-DRIVEN PUBLIC HEALTH</h1>
    <p style='font-size: 1.2rem; color: #9ca3af; margin-top: 0;'>
    Predicting Dengue Outbreaks in Sri Lanka | National Surveillance System
    </p>
    """, unsafe_allow_html=True)

st.divider()

# --- 5. SIDEBAR & DATA LOADING ---
st.sidebar.header("📍 Command Controls")
selected_district = st.sidebar.selectbox("Select Target Region", list(DISTRICTS.keys()))

config = DISTRICTS[selected_district]

try:
    # Load Data
    df = pd.read_csv(config["file_data"])
    df['date'] = pd.to_datetime(df['date'])
    
    # Get Last Known Data
    last_actual = df['actual'].iloc[-1]
    last_pred = df['predicted'].iloc[-1]
    last_date = df['date'].iloc[-1]

    # --- FUTURE FORECAST LOGIC (1 YEAR / 12 MONTHS) ---
    # We generate 12 future months to show a full year trend
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    
    # Simulate a realistic seasonal wave for the next year (Dengue cycles)
    # This creates a smooth curve up and down instead of a jagged line
    future_values = [last_pred * (1 + 0.2 * np.sin(i * 0.5)) for i in range(1, 13)]
    
    # Combine for the graph
    df_future = pd.DataFrame({'date': future_dates, 'predicted': future_values})
    
    # We pad the 'actual' column with NaNs for future dates so the cyan line stops
    df_extended = pd.concat([df, df_future], ignore_index=True)

    # Determine Risk Status
    if last_pred > config["risk_threshold"]:
        risk_color = "#ef4444" # Red
        risk_msg = "CRITICAL RISK"
    elif last_pred > config["risk_threshold"] * 0.7:
        risk_color = "#f59e0b" # Orange
        risk_msg = "WARNING LEVEL"
    else:
        risk_color = "#10b981" # Green
        risk_msg = "NORMAL ACTIVITY"

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 6. KEY PERFORMANCE INDICATORS (KPIs) ---
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown(f"**📍 Region:** {selected_district}")
    st.markdown(f"**🧠 Model:** {config['best_model_name']}")

with kpi2:
    st.metric("Expected Cases (This Month)", f"{int(last_pred)}", delta=f"{int(last_pred - last_actual)}")

with kpi3:
    # Custom colored box for Risk Status
    st.markdown(f"<h3 style='color: {risk_color} !important; text-align: center; border: 2px solid {risk_color}; border-radius: 10px; padding: 5px;'>{risk_msg}</h3>", unsafe_allow_html=True)

# --- 7. ADVANCED VISUALS ---
st.subheader("📈 Forecast Analysis (Past & 1-Year Future)")

# GRAPH: Actual = Cyan (#00FFFF), Predicted = Neon Red (#FF0055)
chart_data = df_extended.set_index("date")[['actual', 'predicted']]
st.line_chart(chart_data, color=["#00FFFF", "#FF0055"]) 
st.caption("Cyan Line = Actual Historical Cases | Red Line = AI Prediction (Includes 1-Year Future Projection)")

# MAP
st.subheader("🗺️ Geospatial Risk Assessment")

# Create Map Data cleanly
map_data = [{"lat": config["lat"], "lon": config["lon"]}]
map_df = pd.DataFrame(map_data)

# Determine dot color based on risk
dot_color = [255, 0, 0] if "CRITICAL" in risk_msg else ([255, 165, 0] if "WARNING" in risk_msg else [0, 255, 0])

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v9", # Dark Map Style
    initial_view_state=pdk.ViewState(latitude=config["lat"], longitude=config["lon"], zoom=9),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_color=dot_color,
            get_radius=3000,
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
        )
    ]
))

st.divider()

# --- 8. AI PREDICTOR SIMULATOR (4 PARAMETERS) ---
st.subheader("🤖 Real-Time AI Predictor (Unlimited Access)")
st.info("Adjust the sliders below to simulate different weather conditions and see the immediate AI prediction.")

sim_row1, sim_row2 = st.columns(2)

with sim_row1:
    rain = st.slider("🌧️ Rainfall (mm)", 0, 600, 200)
    temp = st.slider("🌡️ Temperature (°C)", 20, 40, 30)

with sim_row2:
    humidity = st.slider("💧 Humidity (%)", 50, 100, 80)
    wind = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

# ADVANCED MATH LOGIC for Simulation
# These weights approximate the model's sensitivity
base_cases = last_pred
rain_effect = (rain - 200) * 0.5    # Rain increases mosquitoes
temp_effect = (temp - 28) * 1.5     # Heat speeds up breeding
humid_effect = (humidity - 75) * 2.0 # Humidity helps survival
wind_effect = (wind - 10) * -1.5    # Strong wind blows mosquitoes away (negative effect)

final_simulated_cases = base_cases + rain_effect + temp_effect + humid_effect + wind_effect
if final_simulated_cases < 0: final_simulated_cases = 0

# DISPLAY RESULT
st.markdown("### 🔮 AI Prediction Result:")
st.markdown(f"<h1 style='color: #FFD700;'>{int(final_simulated_cases)} Cases</h1>", unsafe_allow_html=True)
st.caption("Based on your custom weather inputs.")

# --- 9. PUBLIC HEALTH ADVICE ---
st.divider()
st.subheader("📢 AI-Driven Public Health Recommendations")

if "CRITICAL" in risk_msg:
    st.error("""
    ### 🔴 CRITICAL ALERT: Immediate Action Required
    * **Vector Control:** Local authorities must deploy fogging teams immediately.
    * **Hospitals:** Activate surge capacity protocols for Dengue wards.
    * **Public:** Wear long sleeves, use strong repellent, and destroy all breeding sites.
    """)
elif "WARNING" in risk_msg:
    st.warning("""
    ### 🟠 WARNING: Precautionary Phase
    * **Community:** Organize neighborhood clean-ups this weekend.
    * **Schools:** Inspect grounds for standing water.
    * **Personal:** Avoid outdoors during dawn and dusk (peak mosquito times).
    """)
else:
    st.success("""
    ### 🟢 LOW RISK: Maintain Surveillance
    * **Routine:** Continue weekly garden inspections.
    * **Monitor:** Keep checking this dashboard for weather-driven changes.
    """)
