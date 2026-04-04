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


"""
 Walk-forward (a.k.a. expanding window) ARIMA forecast.
 Inputs:
   price_data: DataFrame containing at least a 'Close' column (float-like), indexed by date.
 Output:
   (rmse, mae, r2): error metrics computed on the final 20% of the series.
Refits ARIMA from scratch at every step in test data
"""
def arima_model_forecast(price_data):
    # create copy of data frame to avoid modifying original data
    data = price_data.copy()

    # error handling
    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    # Extract close price data
    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    # ARIMA expects numeric input
    close_price = close_price.astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)
    train_data = list(close_price.iloc[:train_size])
    test_data = close_price.iloc[train_size:]

    predictions = []

    # Walk-forward evaluation:
    for t in range(len(test_data)):
        # using 1,1,1 as order
        model = ARIMA(train_data, order=(1, 1, 1)).fit()
        output = model.forecast()

        # first element is the point forecast
        yhat = output[0]
        predictions.append(yhat)

        # Add the current real observation so the next iteration trains on more data
        actual_value = test_data.iloc[t]
        train_data.append(actual_value)

    # Evaluate on the held-out test segment
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    # Plot predicted series aligned to test_data dates
    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))

    # Prints the summary for the *last* model fitted in the loop
    print("ARIMA model summary: ", model.summary())

    return rmse, mae, r2


"""
 Walk-forward ARIMAX (ARIMA with exogenous regressors) forecast.
 This uses OHLCV features as exogenous variables to help predict 'Close'.
 Exogenous rows used for forecasting must align with the forecast step’s timestamp.
"""
def arima_model_exog_forecast(price_data):
    # create copy of data frame to avoid modifying original data
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # Exogenous (independent) features used to predict the dependent variable 'Close'
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any dependent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)

    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    # These are used to provide the “next” exog row during forecasting
    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    predictions = []

    # rolling forecast using ARIMAX
    for t in range(len(test_close)):
        current_train_exog = exog_data.iloc[:train_size + t]

        model = ARIMA(
            train_close,
            exog=current_train_exog,
            order=(1, 1, 1)
        ).fit()

        # Forecast 1 step ahead using the exog features of that next day
        next_exog = test_exog.iloc[[t]]
        output = model.forecast(steps=1, exog=next_exog)

        # forecast() here returns a pandas series
        yhat = output.iloc[0]
        predictions.append(yhat)

        # expand the training set
        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


"""
 Walk-forward ARIMAX forecast with *post-hoc* sentiment adjustment.
 Predict close with ARIMAX using OHLCV.
 If there is sentiment for that prediction date, adjust the prediction by
 a small amount proportional to (sentiment_score * predicted_price).
 The sentiment score can be lagged (shifted) by sentiment_lag days.
"""
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

    # Convert timestamps and aggregate multiple articles per calendar day.
    news_data["datetime"] = pd.to_datetime(news_data["datetime"])
    news_data = news_data.set_index("datetime")

    # normalize() drops intraday time and keeps only the date portion
    news_data.index = news_data.index.normalize()

    # Average sentiment per day across all retrieved articles
    average_sentiment = news_data["sentiment_score"].groupby(news_data.index).mean()

    # Lag sentiment by N days so only use sentiment that would have been known earlier
    sentiment_lag = 3
    average_sentiment_lagged = average_sentiment.shift(sentiment_lag)

    # Normalised dates for the price series. Used to align sentiment with prices
    price_dates = pd.to_datetime(price_data.index).normalize()

    # Exogenous features (OHLCV)
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
        current_train_exog = exog_data.iloc[:train_size + t]

        model = ARIMA(
            train_close,
            exog=current_train_exog,
            order=(1, 1, 1)
        ).fit()

        next_exog = test_exog.iloc[[t]]
        output = model.forecast(steps=1, exog=next_exog)
        yhat = output.iloc[0]

        # Map the prediction in the test window to its calendar date
        current_date = price_dates[train_size + t]

        # Apply adjustment only if we have sentiment for that date.
        if current_date in average_sentiment.index:
            sentiment_score = float(average_sentiment_lagged.loc[current_date])

            # A scaling factor is used so sentiment doesn't dominate price.
            sentiment_adjustment = sentiment_score * yhat * 0.001
            prediction = yhat + sentiment_adjustment
        else:
            prediction = yhat

        predictions.append(prediction)

        # Expand training with the true observed value
        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(price_data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


# Plot ACF/PACF of the differenced training series.
# This helps graphically identify potential AR (p) and MA (q) orders for ARIMA.
def plot_autocorrelation(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]

    # diff() is used because ARIMA typically requires (approx.) stationary input
    plot_acf(close_price.diff().dropna())
    plt.show()

    plot_pacf(close_price.diff().dropna())
    plt.show()


# Run ADF test on differenced data.
def adf_test(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]

    result = adfuller(close_price.diff().dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")


# Use pmdarima to suggest the differencing order d
def determine_differencing(price_data):
    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]

    ndiffs(close_price, test="adf")

# Decompose the training series into trend, seasonal, and residual components.
def decompose_time_series(price_data):

    data = price_data.copy()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    close_price = train_data["Close"]

    decomposition = seasonal_decompose(close_price, model="additive")
    decomposition.plot()
    plt.show()
