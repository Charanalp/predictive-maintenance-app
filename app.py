import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Custom CSS for Modern Gradient & Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        font-size: 22px;
        text-align: center;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 20px;
    }
    .stSidebar > div:first-child {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > input, .stNumberInput > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        font-weight: bold;
    }
    .stButton > button {
        background-color: #ff4b2b;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown('<div class="title">Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Predict Machine Failure using Real-time Sensor Data</div>', unsafe_allow_html=True)

# Load Model & Scaler
model = joblib.load('predictive_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar - Sensor Inputs
st.sidebar.markdown('<div class="section-title">🔧 Sensor Inputs</div>', unsafe_allow_html=True)

air_temp = st.sidebar.number_input('**Air Temperature (K)**', min_value=290.0, max_value=320.0, value=300.0)
process_temp = st.sidebar.number_input('**Process Temperature (K)**', min_value=290.0, max_value=320.0, value=305.0)
rpm = st.sidebar.number_input('**Rotational Speed (rpm)**', min_value=1000.0, max_value=3000.0, value=1500.0)
torque = st.sidebar.number_input('**Torque (Nm)**', min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.sidebar.number_input('**Tool Wear (min)**', min_value=0.0, max_value=300.0, value=100.0)

# Predict Button
if st.sidebar.button('Predict Failure'):
    input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result = '⚠️ Machine Failure Detected!' if prediction[0] == 1 else '✅ Machine is Operating Normally.'

    st.markdown(f'<div class="section-title">{result}</div>', unsafe_allow_html=True)
