import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Page Configuration
st.set_page_config(
    page_title="Heavy-Duty Truck Sales Time Series Forecasting",
    page_icon="🚛",
    layout="wide"
)

# Header & Image
st.title("🚛 Heavy-Duty Truck Sales Time Series Forecasting")
st.markdown("This web application analyzes historical heavy-duty truck sales data and performs future sales forecasting using the **SARIMAX** model.")

st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIAfxqxBeIo4NbxFADPRbTMlzK1Qc_agH_f3EM1Msi1r7EAPmwpCSqNxQ&s=10",
    width=750
)

st.divider()

# Data Loading Function
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('Truck_sales.csv')
    
    # Convert date format and set as index (Month-Year -> e.g., 03-Jan)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%y-%b')
    df = df.set_index('Month-Year').asfreq('MS')
    return df

# Load Data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset. Make sure 'Truck_sales.csv' is in the same directory. Error: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.header("⚙️ Settings & Menu")
menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview & EDA", "Time Series Plot", "SARIMAX Forecasting"]
)

# SECTION 1: EDA
if menu == "Dataset Overview & EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        st.subheader("Last 5 Rows")
        st.dataframe(df.tail())

    with col2:
        st.subheader("Statistical Summary (Describe)")
        st.dataframe(df.describe())
        
        st.subheader("Dataset Information")
        st.write(f"**Total Observations:** {df.shape[0]}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

# SECTION 2: TIME SERIES PLOT
elif menu == "Time Series Plot":
    st.header("📈 Heavy-Duty Truck Sales Over Time")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Number_Trucks_Sold'], marker='o', color='tab:blue', linestyle='-')
    ax.set_title("Monthly Truck Sales Trend", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Trucks Sold", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)

    # ADF Test
    st.subheader("Augmented Dickey-Fuller Stationarity Test (ADF)")
    if st.button("Run ADF Test"):
        result = adfuller(df['Number_Trucks_Sold'].dropna())
        st.write(f"**ADF Statistic:** {result[0]:.4f}")
        st.write(f"**p-value:** {result[1]:.4f}")
        if result[1] <= 0.05:
            st.success("The time series is stationary (p <= 0.05).")
        else:
            st.warning("The time series is non-stationary (p > 0.05). Differencing may be required.")

# SECTION 3: SARIMAX FORECASTING
elif menu == "SARIMAX Forecasting":
    st.header("🔮 Sales Forecasting with SARIMAX")
    
    st.sidebar.subheader("Model Parameters (p, d, q)")
    p = st.sidebar.number_input("p (AR)", min_value=0, max_value=5, value=1)
    d = st.sidebar.number_input("d (I)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("q (MA)", min_value=0, max_value=5, value=1)
    
    st.sidebar.subheader("Seasonal Parameters (P, D, Q, s)")
    P = st.sidebar.number_input("P (Seasonal AR)", min_value=0, max_value=5, value=1)
    D = st.sidebar.number_input("D (Seasonal I)", min_value=0, max_value=2, value=1)
    Q = st.sidebar.number_input("Q (Seasonal MA)", min_value=0, max_value=5, value=1)
    s = st.sidebar.number_input("s (Seasonal Period)", min_value=1, max_value=24, value=12)
    
    forecast_steps = st.slider("Select Forecast Horizon (Months):", min_value=1, max_value=36, value=12)

    if st.button("Train Model & Forecast"):
        with st.spinner("Training SARIMAX model..."):
            try:
                # SARIMAX Model Setup
                model = SARIMAX(
                    df['Number_Trucks_Sold'],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s)
                )
                results = model.fit(disp=False)
                
                # Forecasting
                forecast = results.get_forecast(steps=forecast_steps)
                forecast_df = forecast.conf_int()
                forecast_df['Forecast'] = forecast.predicted_mean
                
                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df['Number_Trucks_Sold'], label='Historical Sales', color='black')
                ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
                ax.fill_between(
                    forecast_df.index,
                    forecast_df.iloc[:, 0],
                    forecast_df.iloc[:, 1],
                    color='pink',
                    alpha=0.3,
                    label='Confidence Interval'
                )
                ax.set_title("Truck Sales Forecast", fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Sales Quantity", fontsize=12)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                
                # Forecast Table
                st.subheader("📋 Forecast Data Table")
                st.dataframe(forecast_df[['Forecast']].rename(columns={'Forecast': 'Forecasted Sales'}))
                
            except Exception as e:
                st.error(f"An error occurred while training the model: {e}")