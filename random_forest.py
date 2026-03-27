from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from data_visualisation import plot_actual_vs_predicted
import numpy as np


def rf_baseline_forecast(price_data):
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

    for t in range(len(test_close)):
        # dependent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        model = RandomForestRegressor(random_state=42).fit(current_train_exog, current_train_close)

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        yhat = model.predict(next_exog)[0]
        predictions.append(yhat)

        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


def rf_tuned_forecast(price_data):
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

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(train_exog, train_close)
    best_params = grid_search.best_params_
    print("Best Hyperparameters: ", best_params)

    predictions = []

    for t in range(len(test_close)):
        # dependent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        model = (RandomForestRegressor(**best_params, random_state=42))
        model.fit(current_train_exog, current_train_close)

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        yhat = model.predict(next_exog)[0]
        predictions.append(yhat)

        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


def rf_sentiment_forecast(price_data, news_sentiment):
    data = price_data.copy()
    news_data = news_sentiment.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # aggregate sentiment scores by date
    news_data["datetime"] = pd.to_datetime(news_data["datetime"])
    news_data = news_data.set_index("datetime")
    news_data.index = news_data.index.normalize()

    average_sentiment = news_data["sentiment_score"].groupby(news_data.index).mean()

    sentiment_lag = 2
    average_sentiment_lagged = average_sentiment.shift(sentiment_lag)

    price_dates = pd.to_datetime(price_data.index).normalize()

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

    best_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    }

    predictions = []

    for t in range(len(test_close)):
        # dependent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        model = (RandomForestRegressor(**best_params, random_state=42))
        model.fit(current_train_exog, current_train_close)

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        yhat = model.predict(next_exog)[0]

        current_date = price_dates[train_size + t]

        if current_date in average_sentiment.index:
            #sentiment_score = average_sentiment[current_date]
            sentiment_score = float(average_sentiment_lagged.loc[current_date])
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

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


def rf_step_forecast(price_data):
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

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(train_exog, train_close)
    best_params = grid_search.best_params_
    print("Best Hyperparameters: ", best_params)

    #predictions = []
    model = RandomForestRegressor(random_state=42).fit(train_exog, train_close)
    predictions = model.predict(test_exog)
    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2


    """
    #predictions = []
    model = RandomForestRegressor(random_state=42).fit(train_exog, train_close)
    predictions = model.predict(test_exog)
    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    mae = mean_absolute_error(test_close, predictions)
    r2 = r2_score(test_close, predictions)

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": predictions}, index=test_close.index))

    return rmse, mae, r2
    """

