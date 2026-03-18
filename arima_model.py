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


def arima_model_exog_forecast(price_data):
    # create copy of data frame to avoid modifying original data
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # define dependent features
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any dependent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)

    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    predictions = []
    # rolling forecast using ARIMAX
    for t in range(len(test_close)):
        # dependent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]

        model = ARIMA(
            train_close,
            exog=current_train_exog,
            order=(1, 1, 1)
        ).fit()

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        output = model.forecast(steps=1, exog=next_exog)
        yhat = output.iloc[0]
        predictions.append(yhat)

        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


def arima_model_sentiment_forecast(price_data, news_sentiment):
    # create copy of data frame to avoid modifying original data
    price_data = price_data.copy()
    news_data = news_sentiment.copy()

    if "Close" not in price_data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = price_data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # aggregate sentiment scores by date
    news_data["datetime"] = pd.to_datetime(news_data["datetime"])
    news_data = news_data.set_index("datetime")
    news_data.index = news_data.index.normalize()

    average_sentiment = news_data["sentiment_score"].groupby(news_data.index).mean()
    price_dates = pd.to_datetime(price_data.index).normalize()

    # define dependent features
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in price_data.columns]

    if not available_features:
        raise ValueError("Data does not contain any dependent features (Open, High, Low, Volume)")

    exog_data = price_data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)

    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    predictions = []
    # rolling forecast using ARIMAX
    for t in range(len(test_close)):
        # dependent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]

        model = ARIMA(
            train_close,
            exog=current_train_exog,
            order=(1, 1, 1)
        ).fit()

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        output = model.forecast(steps=1, exog=next_exog)
        yhat = output.iloc[0]
        #predictions.append(yhat)

        current_date = price_dates[train_size + t]

        if current_date in average_sentiment.index:
            sentiment_score = average_sentiment[current_date]
            sentiment_adjustment = sentiment_score * yhat * 0.01
            prediction = yhat + sentiment_adjustment
        else:
            prediction = yhat

        predictions.append(prediction)
        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(price_data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

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
