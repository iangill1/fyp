import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data_retrieval import price_retrieval

def naive_model_forecast(price_data):
    #download historical stock data for a company using yfinance api
    data = price_data

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    #create copy of data frame to avoid modifying original data
    data = data.copy()
    #shift the close column down by 1 so that each days forecast is yesterdays value
    data['Naive Forecast'] = data['Close'].shift(1)
    #this means first value will be NaN, so drop it
    data = data.dropna()

    #calculate rmse and mae
    rmse = np.sqrt(mean_squared_error(data["Close"], data["Naive Forecast"]))
    mae = mean_absolute_error(data["Close"], data["Naive Forecast"])

    return rmse, mae

"""
#plot results
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Actual')
    plt.plot(data['Naive Forecast'], label='Naive Forecast', linestyle='--')
    plt.legend()
    plt.title('Naive Model Forecast vs Actual')
    plt.show()
"""