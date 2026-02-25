import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_visualisation import plot_actual_vs_predicted
from sentiment_analysis import analyse_sentiment


def naive_model_forecast(price_data):
    # create copy of data frame
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
    close_price = close_price.astype(float)

    # split data
    train_size = int(len(close_price) * 0.8)
    train_data = close_price.iloc[:train_size]
    test_data = close_price.iloc[train_size:]

    history = train_data.tolist()
    predictions = []
    for t in range(len(test_data)):
        prediction = history[-1]  # naive forecast is just the last observed value
        predictions.append(prediction)
        history.append(test_data.iloc[t])  # add the actual value to history for next prediction

    # calculate rmse and mae
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))

    return rmse, mae, r2


def naive_sentiment_forecast(price_data, news_sentiment):
    # create copy of data frame
    price_data = price_data.copy()
    news_data = news_sentiment.copy()

    if "Close" not in price_data.columns:
        raise ValueError("Data does not contain 'Close' column")
    if "sentiment_score" not in news_sentiment.columns:
        raise ValueError("Data does not contain 'sentiment_score' column")

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

    # split data
    train_size = int(len(close_price) * 0.8)
    train_data = close_price.iloc[:train_size]
    test_data = close_price.iloc[train_size:]

    history = train_data.tolist()
    predictions = []
    for t in range(len(test_data)):
        last_price = history[-1]  # naive forecast is just the last observed value
        current_date = price_dates[train_size + t]

        if current_date in average_sentiment.index:
            sentiment_score = average_sentiment[current_date]
            sentiment_adjustment = sentiment_score * last_price * 0.05
            prediction = last_price + sentiment_adjustment
        else:
            prediction = last_price

        predictions.append(prediction)
        history.append(test_data.iloc[t])  # add the actual value to history for next prediction

    # calculate rmse and mae
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)

    plot_actual_vs_predicted(price_data, pd.DataFrame({"Predicted": predictions}, index=test_data.index))

    return rmse, mae, r2
