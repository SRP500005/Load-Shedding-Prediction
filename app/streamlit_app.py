import streamlit as st

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import (
    predict_load_shedding
)


st.set_page_config(
    page_title="Load Shedding Prediction",
    page_icon="⚡",
    layout="wide"
)


st.title(
    "⚡ Load Shedding Prediction System"
)

st.write(
    """
    Machine Learning application for predicting
    the probability of electricity load shedding
    using grid demand, generation, weather and
    market conditions.
    """
)

st.info(
    "The current model was trained on synthetic hourly grid data from 2020–2024."
)


# ==========================================
# DATE / TIME FEATURES
# ==========================================

st.subheader(
    "1. Time Information"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    year = st.number_input(
        "Year",
        min_value=2020,
        max_value=2035,
        value=2024
    )

with col2:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

with col3:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )

with col4:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=18
    )


col1, col2, col3 = st.columns(3)

with col1:

    day_of_week = st.number_input(
        "Day of Week (0=Monday)",
        min_value=0,
        max_value=6,
        value=2
    )

with col2:

    day_of_year = st.number_input(
        "Day of Year",
        min_value=1,
        max_value=366,
        value=167
    )

with col3:

    season = st.selectbox(
        "Season",
        [
            "Spring",
            "Summer",
            "Autumn",
            "Winter"
        ]
    )


is_weekend = st.checkbox(
    "Weekend"
)

is_holiday = st.checkbox(
    "Holiday"
)


# ==========================================
# WEATHER
# ==========================================

st.subheader(
    "2. Weather Conditions"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        value=20.0
    )

with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with col3:

    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        value=5.0
    )

with col4:

    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )


solar_irradiance = st.number_input(
    "Solar Irradiance (W/m²)",
    min_value=0.0,
    value=400.0
)


# ==========================================
# LOAD
# ==========================================

st.subheader(
    "3. Electricity Demand"
)

col1, col2, col3 = st.columns(3)

with col1:

    industrial_load = st.number_input(
        "Industrial Load (MW)",
        min_value=0.0,
        value=200.0
    )

with col2:

    residential_load = st.number_input(
        "Residential Load (MW)",
        min_value=0.0,
        value=250.0
    )

with col3:

    commercial_load = st.number_input(
        "Commercial Load (MW)",
        min_value=0.0,
        value=150.0
    )


total_demand = st.number_input(
    "Total Demand (MW)",
    min_value=0.0,
    value=600.0
)


col1, col2, col3 = st.columns(3)

with col1:

    demand_lag_24h = st.number_input(
        "Demand 24 Hours Ago (MW)",
        min_value=0.0,
        value=590.0
    )

with col2:

    demand_lag_168h = st.number_input(
        "Demand 1 Week Ago (MW)",
        min_value=0.0,
        value=580.0
    )

with col3:

    demand_rolling = st.number_input(
        "24h Average Demand (MW)",
        min_value=0.0,
        value=585.0
    )


# ==========================================
# GENERATION
# ==========================================

st.subheader(
    "4. Power Generation"
)

col1, col2, col3 = st.columns(3)

with col1:

    solar_generation = st.number_input(
        "Solar Generation (MW)",
        min_value=0.0,
        value=100.0
    )

with col2:

    wind_generation = st.number_input(
        "Wind Generation (MW)",
        min_value=0.0,
        value=120.0
    )

with col3:

    renewable_generation = st.number_input(
        "Total Renewable Generation (MW)",
        min_value=0.0,
        value=220.0
    )


conventional_generation = st.number_input(
    "Conventional Generation (MW)",
    min_value=0.0,
    value=450.0
)


# ==========================================
# GRID
# ==========================================

st.subheader(
    "5. Grid Conditions"
)

col1, col2, col3 = st.columns(3)

with col1:

    grid_frequency = st.number_input(
        "Grid Frequency (Hz)",
        value=50.0,
        format="%.3f"
    )

with col2:

    reserve_margin = st.number_input(
        "Reserve Margin (%)",
        value=35.0
    )

with col3:

    criticality_score = st.number_input(
        "Criticality Score",
        min_value=0.0,
        value=0.5
    )


feeder_id = st.selectbox(
    "Feeder",
    [
        f"FDR_{i:02d}"
        for i in range(1, 11)
    ]
)


# ==========================================
# MARKET
# ==========================================

st.subheader(
    "6. Energy Market Conditions"
)

col1, col2, col3 = st.columns(3)

with col1:

    gas_price = st.number_input(
        "Gas Price (€/MWh)",
        min_value=0.0,
        value=40.0
    )

with col2:

    coal_price = st.number_input(
        "Coal Price (€/MWh)",
        min_value=0.0,
        value=20.0
    )

with col3:

    carbon_price = st.number_input(
        "Carbon Price (€/ton)",
        min_value=0.0,
        value=80.0
    )


col1, col2 = st.columns(2)

with col1:

    electricity_price = st.number_input(
        "Electricity Price (€/MWh)",
        value=100.0
    )

with col2:

    price_volatility = st.number_input(
        "Price Volatility Index",
        min_value=0.0,
        value=1.0
    )


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "Predict Load Shedding Risk",
    type="primary"
):

    input_data = {

        "year": year,
        "month": month,
        "day": day,
        "hour": hour,

        "day_of_week": day_of_week,
        "day_of_year": day_of_year,

        "is_weekend": int(
            is_weekend
        ),

        "is_holiday": int(
            is_holiday
        ),

        "season": season,

        "temperature_C": temperature,
        "humidity_pct": humidity,
        "wind_speed_ms": wind_speed,
        "cloud_cover_pct": cloud_cover,

        "solar_irradiance_wm2":
            solar_irradiance,

        "industrial_load_mw":
            industrial_load,

        "residential_load_mw":
            residential_load,

        "commercial_load_mw":
            commercial_load,

        "total_demand_mw":
            total_demand,

        "demand_lag_24h":
            demand_lag_24h,

        "demand_lag_168h":
            demand_lag_168h,

        "demand_rolling_mean_24h":
            demand_rolling,

        "solar_generation_mw":
            solar_generation,

        "wind_generation_mw":
            wind_generation,

        "renewable_generation_mw":
            renewable_generation,

        "conventional_generation_mw":
            conventional_generation,

        "grid_frequency_hz":
            grid_frequency,

        "reserve_margin_pct":
            reserve_margin,

        "fuel_price_gas_eur_mwh":
            gas_price,

        "fuel_price_coal_eur_mwh":
            coal_price,

        "carbon_price_eur_ton":
            carbon_price,

        "electricity_price_eur_mwh":
            electricity_price,

        "price_volatility_index":
            price_volatility,

        "feeder_id":
            feeder_id,

        "criticality_score":
            criticality_score
    }


    result = predict_load_shedding(
        input_data
    )


    probability = result[
        "probability"
    ]

    prediction = result[
        "prediction"
    ]


    st.divider()

    st.subheader(
        "Prediction Result"
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Load Shedding Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Classification Threshold",
            f"{result['threshold']:.2f}"
        )


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK: Load shedding predicted."
        )

    else:

        st.success(
            "✅ LOW RISK: No load shedding predicted."
        )


    if probability >= 0.75:

        risk_level = "Very High"

    elif probability >= 0.50:

        risk_level = "High"

    elif probability >= 0.25:

        risk_level = "Moderate"

    else:

        risk_level = "Low"


    st.write(
        f"**Risk Level:** {risk_level}"
    )


    st.write(
        f"**Total Demand:** {total_demand:.2f} MW"
    )

    st.write(
        f"**Reserve Margin:** {reserve_margin:.2f}%"
    )

    st.write(
        f"**Grid Frequency:** {grid_frequency:.3f} Hz"
    )


st.divider()

st.caption(
    "Portfolio project — Load Shedding Prediction using Machine Learning and Explainable AI."
)