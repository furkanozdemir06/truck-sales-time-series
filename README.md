# 🚛 Heavy-Duty Truck Sales Forecasting

A time series forecasting project that analyzes monthly heavy-duty truck sales and predicts future demand using **SARIMAX** and **Prophet**.

## 📌 Project Overview

The project uses historical truck sales data to:

- Analyze trend and seasonality
- Test stationarity with the ADF test
- Build SARIMAX forecasts
- Compare results with Prophet
- Provide an interactive Streamlit dashboard

## 📊 Dataset

The dataset contains **144 monthly observations** from **January 2003 to December 2014**.

Main columns:

- `Month-Year`
- `Number_Trucks_Sold`

No missing values are reported.

## 🔎 Time Series Analysis

The notebook includes:

- Historical sales visualization
- Trend and seasonal analysis
- ADF stationarity test
- SARIMAX forecasting
- Prophet forecasting

ADF result:

```text
ADF Statistic: 1.1159
p-value: 0.9954
```

The original series is considered **non-stationary**.

## 🔮 SARIMAX Model

```python
SARIMAX(
    df["Number_Trucks_Sold"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
```

The model generates a **12-month forecast** while accounting for trend and yearly seasonality.

## 📈 Prophet

Prophet is also used as an alternative forecasting model:

```python
Prophet(yearly_seasonality=True)
```

## 🖥️ Streamlit Dashboard

The `trucks.py` application includes:

- Dataset overview
- Historical sales chart
- ADF stationarity test
- Custom SARIMAX parameters
- 1–36 month forecast horizon
- Forecast confidence intervals
- Forecast results table

Run the app:

```bash
streamlit run trucks.py
```

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- SARIMAX
- Prophet
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
truck-sales-time-series/
├── TimeSeriesTrucks.ipynb
├── trucks.py
├── Truck_sales.csv
└── README.md
```

## 🎯 Skills Demonstrated

- Time series analysis
- Stationarity testing
- Trend and seasonality analysis
- SARIMAX modeling
- Prophet forecasting
- Streamlit dashboard development

---

Built with Python, Statsmodels, Prophet, and Streamlit. 🚛📈
