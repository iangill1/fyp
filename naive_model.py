import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#download historical stock data for a company using yfinance api
data = yf.download("ttwo", interval="60m", start=("2025-11-05"), auto_adjust=False)

print(data.tail())

#shift the close column down by 1 so that each days forecast is yesterdays value
data['Naive Forecast'] = data['Close'].shift(1)
#this means first value will be NaN, so drop it
data = data.dropna()

#calculate rmse and mae
rmse = np.sqrt(mean_squared_error(data['Close'], data['Naive Forecast']))
mae = mean_absolute_error(data['Close'], data['Naive Forecast'])

#print results
print("rmse: ", rmse)
print("mae: ", mae)

#plot results
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Actual')
plt.plot(data['Naive Forecast'], label='Naive Forecast', linestyle='--')
plt.legend()
plt.title('Naive Model Forecast vs Actual')
plt.show()
