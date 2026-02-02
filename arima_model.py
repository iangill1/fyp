import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs


def arima_model_forecast(price_data):
    #create copy of data frame to avoid modifying original data
    data = price_data.copy()
    data1 = data[["Close"]].copy()
    data2 = data1.describe()
    print("Describe", data2)

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    #fit ARIMA model
    model = ARIMA(data["Close"], order=(6, 1, 3))
    arima_prediction = model.fit()
    print(arima_prediction.summary())
    #get predictions
    rmse = np.sqrt(mean_squared_error(data["Close"], arima_prediction.fittedvalues))
    mae = mean_absolute_error(data["Close"], arima_prediction.fittedvalues)

    return rmse, mae


def plot_autocorrelation(price_data):
    plot_acf(price_data["Close"].diff().dropna())
    plt.show()
    #plot_pacf(price_data['Close'])
    #plt.show()


def adf_test(price_data):
    df = price_data.copy()
    close_price = df["Close"]
    result = adfuller(close_price.diff().dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")


def determine_differencing(price_data):
    close_price = price_data["Close"].copy()
    ndiffs(close_price, test="adf")
