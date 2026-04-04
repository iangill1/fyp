import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_visualisation import plot_actual_vs_predicted


"""
Naive baseline forecast
Predict tomorrow’s close as today’s close.
Input: price_data
Output: rmse, mae, r2
"""
def naive_model_forecast(price_data):
    # create copy of data frame
    data = price_data.copy()

    # Basic validation
    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    # Extract Close price
    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
    close_price = close_price.astype(float)

    # Train/test split (80% train, 20% test)
    train_size = int(len(close_price) * 0.8)
    train_data = close_price.iloc[:train_size]
    test_data = close_price.iloc[train_size:]

    # 'history' is the growing list of observed closes. Append each true test value
    history = train_data.tolist()
    predictions = []

    for t in range(len(test_data)):
        # Naive forecast = last observed close
        prediction = history[-1]
        predictions.append(prediction)

        # Reveal the true close and add it to history for the next step
        history.append(test_data.iloc[t])

    # Error metrics on the held-out test segment
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    # Plot predicted series aligned to the test dates
    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))

    return rmse, mae, r2


"""
Sentiment-incorporated naive forecast.
Start with the naive prediction (last observed close), then apply a small adjustment based on the average news sentiment for that date.
"""

def naive_sentiment_forecast(price_data, news_sentiment):
    price_data = price_data.copy()
    news_data = news_sentiment.copy()

    # Validation
    if "Close" not in price_data.columns:
        raise ValueError("Data does not contain 'Close' column")
    if "sentiment_score" not in news_sentiment.columns:
        raise ValueError("Data does not contain 'sentiment_score' column")

    close_price = price_data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # Convert timestamps, set as index, and normalise to calendar dates (drop time-of-day)
    news_data["datetime"] = pd.to_datetime(news_data["datetime"])
    news_data = news_data.set_index("datetime")
    news_data.index = news_data.index.normalize()

    # Average sentiment score per day
    average_sentiment = news_data["sentiment_score"].groupby(news_data.index).mean()

    # Lag sentiment by N days (0 = same-day sentiment)
    sentiment_lag = 0
    average_sentiment_lagged = average_sentiment.shift(sentiment_lag)

    # Normalised price dates to align day-level sentiment with price rows.
    price_dates = pd.DatetimeIndex(pd.to_datetime(price_data.index).normalize())

    # Train/test split (80% train, 20% test)
    train_size = int(len(close_price) * 0.8)
    train_data = close_price.iloc[:train_size]
    test_data = close_price.iloc[train_size:]

    history = train_data.tolist()
    predictions = []

    for t in range(len(test_data)):
        # Base naive prediction
        last_price = history[-1]

        # Date corresponding to this test point
        current_date = price_dates[train_size + t]

        # If we have sentiment for this date, apply a simple adjustment
        if current_date in average_sentiment.index:
            sentiment_score = float(average_sentiment_lagged.loc[current_date])
            sentiment_adjustment = sentiment_score * last_price * 0.01
            prediction = last_price + sentiment_adjustment
        else:
            prediction = last_price

        predictions.append(prediction)

        # Reveal the true close for walk-forward evaluation
        history.append(test_data.iloc[t])

    # Error metrics on the held-out test segment
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    plot_actual_vs_predicted(price_data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))

    return rmse, mae, r2
