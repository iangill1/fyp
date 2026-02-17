import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
import pmdarima as pm
from data_visualisation import plot_actual_vs_predicted


def arima_model_forecast(price_data):
    # create copy of data frame to avoid modifying original data
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)
    train_data = list(close_price.iloc[:train_size])
    test_data = close_price.iloc[train_size:]
    predictions = []
    for t in range(len(test_data)):
        model = ARIMA(train_data, order=(1, 1, 1)).fit()
        output = model.forecast()
        yhat = output[0]
        predictions.append(yhat)
        actual_value = test_data.iloc[t]
        train_data.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))
    print("ARIMA model summary: ", model.summary())

    return rmse, mae, r2


# plot ACF and PACF on differenced data
def plot_autocorrelation(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]
    plot_acf(close_price.diff().dropna())
    plt.show()
    plot_pacf(close_price.diff().dropna())
    plt.show()


# perform ADF test on differenced data to check for stationarity
def adf_test(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]
    result = adfuller(close_price.diff().dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")


# using pmdarima to determine optimal differencing order
def determine_differencing(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]
    ndiffs(close_price, test="adf")


def decompose_time_series(price_data):

    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]
    decomposition = seasonal_decompose(close_price, model="additive")
    decomposition.plot()
    plt.show()
