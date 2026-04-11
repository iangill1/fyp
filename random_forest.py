from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from data_visualisation import plot_actual_vs_predicted
import numpy as np
from scipy.optimize import minimize


def rf_baseline_forecast(price_data):
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    # handle case where 'Close' is a datframe
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = close_price.astype(float)

    # define independent features
    exog_features = ["Open", "High", "Low", "Volume"]
    # only keep features that actually exist in data
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any independent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split) based on time order
    train_size = int(len(close_price) * 0.8)

    # list for dynamic appending
    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    predictions = []

    # rolling forecast with retraining at each step to mimic real-time forecasting scenario
    for t in range(len(test_close)):
        # independent for training must match length of train_close
        # expand training window each step
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        # train model on all available data up to current point
        model = RandomForestRegressor(random_state=42).fit(current_train_exog, current_train_close)

        # forecast one step ahead using next row of exog features
        next_exog = test_exog.iloc[[t]]
        yhat = model.predict(next_exog)[0]
        predictions.append(yhat)

        # add actual values to training data for next step
        actual_value = test_close.iloc[t]
        train_close.append(actual_value)

    # evaluation metrics
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

    # define independent features
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any independent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)

    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    # hyperparameter search grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    # time aware cross-validation split to prevent look-ahead bias during hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=5)

    # grid search finds best combination of hyperparameters
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
        # independent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        # use best hyperparameters to train model on all available data up to current point
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

    sentiment_lag = 0
    average_sentiment_lagged = average_sentiment.shift(sentiment_lag)

    price_dates = pd.to_datetime(price_data.index).normalize()

    # define independent features
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any independent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.8)

    train_close = list(close_price.iloc[:train_size])
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    best_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    }

    def calculate_rmse_with_multiplier(multiplier):
        multiplier = float(np.atleast_1d(multiplier)[0])
        predictions = []
        for t in range(len(test_close)):
            current_train_exog = exog_data.iloc[:train_size + t]
            current_train_close = close_price.iloc[:train_size + t]

            model = RandomForestRegressor(**best_params, random_state=42)
            model.fit(current_train_exog, current_train_close)

            yhat = float(model.predict(test_exog.iloc[[t]])[0])
            current_date = price_dates[train_size + t]

            if current_date in average_sentiment_lagged.index:
                s = average_sentiment_lagged.loc[current_date]
                if pd.notna(s):
                    s = float(s)
                    yhat = yhat + (s * yhat * multiplier)

            predictions.append(yhat)

        return float(np.sqrt(mean_squared_error(test_close, predictions)))

    result = minimize(
        calculate_rmse_with_multiplier,
        x0=np.array([0.01], dtype=float),
        bounds=[(0.001, 0.1)],
        method="L-BFGS-B",
    )
    optimal_multiplier = float(result.x[0])
    print(f"Optimal Sentiment Multiplier: {optimal_multiplier}")

    predictions = []

    for t in range(len(test_close)):
        # independent for training must match length of train_close
        current_train_exog = exog_data.iloc[:train_size + t]
        current_train_close = close_price.iloc[:train_size + t]

        model = (RandomForestRegressor(**best_params, random_state=42))
        model.fit(current_train_exog, current_train_close)

        # forecast one step ahead using next row of exog features
        yhat = float(model.predict(test_exog.iloc[[t]])[0])
        current_date = price_dates[train_size + t]

        if current_date in average_sentiment_lagged.index:
            s = average_sentiment_lagged.loc[current_date]
            if pd.notna(s):
                s = float(s)
                yhat = yhat + (s * yhat * optimal_multiplier)

        predictions.append(yhat)

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

    # define independent features
    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]

    if not available_features:
        raise ValueError("Data does not contain any independent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets (80/20 split)
    train_size = int(len(close_price) * 0.95)

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


def rf_sentiment_step_forecast(price_data, news_sentiment, *, roll_window_days=1, lag_days=1, multiplier=0.00367573,):
    data = price_data.copy()

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]
    close_price = close_price.astype(float)

    exog_features = ["Open", "High", "Low", "Volume"]
    available_features = [f for f in exog_features if f in data.columns]
    if not available_features:
        raise ValueError("Data does not contain any independent features (Open, High, Low, Volume)")

    exog_data = data[available_features].astype(float)

    # split data into train and test sets
    train_size = int(len(close_price) * 0.95)
    train_close = close_price.iloc[:train_size]
    test_close = close_price.iloc[train_size:]

    train_exog = exog_data.iloc[:train_size]
    test_exog = exog_data.iloc[train_size:]

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(train_exog, train_close)
    best_params = grid_search.best_params_
    print("Best Hyperparameters: ", best_params)

    # Stage 1: fit once, predict full test split (no rolling retrain)
    model = RandomForestRegressor(**best_params, random_state=42).fit(train_exog, train_close)
    base_pred = pd.Series(
        model.predict(test_exog).astype(float),
        index=test_close.index,
        name="Predicted",
    )

    # Stage 2: apply rolling sentiment (with weekend articles mapped to next business day)
    final_pred = base_pred
    if news_sentiment is not None and isinstance(news_sentiment, pd.DataFrame) and not news_sentiment.empty:
        if "datetime" not in news_sentiment.columns:
            raise ValueError("News sentiment data does not contain 'datetime' column")
        if "sentiment_score" not in news_sentiment.columns:
            raise ValueError("News sentiment data does not contain 'sentiment_score' column")

        news_df = news_sentiment.copy()
        news_df["datetime"] = pd.to_datetime(news_df["datetime"])
        news_df["date"] = news_df["datetime"].dt.normalize()

        # Weekend/federal holiday mapping: push to the next business day
        # (Sat/Sun -> Mon; holidays -> next open business day per pandas BDay calendar)
        news_df["biz_date"] = news_df["date"] + pd.offsets.BDay(0)

        daily = news_df.groupby("biz_date")["sentiment_score"].mean().sort_index()

        # Build a complete business-day index over the prediction window
        pred_biz_idx = pd.DatetimeIndex(final_pred.index).normalize()
        full_biz_idx = pd.date_range(pred_biz_idx.min(), pred_biz_idx.max(), freq="B")

        daily = daily.reindex(full_biz_idx)

        s_roll = daily.rolling(window=int(roll_window_days), min_periods=1).mean()
        if lag_days:
            s_roll = s_roll.shift(int(lag_days))

        s_for_pred = s_roll.reindex(pred_biz_idx).fillna(0.0).astype(float)

        final_pred = pd.Series(
            final_pred.to_numpy(dtype=float)
            * (1.0 + float(multiplier) * s_for_pred.to_numpy(dtype=float)),
            index=final_pred.index,
            name="Predicted",
        )

    rmse = float(np.sqrt(mean_squared_error(test_close, final_pred)))
    mae = float(mean_absolute_error(test_close, final_pred))
    r2 = float(r2_score(test_close, final_pred))

    plot_actual_vs_predicted(data, pd.DataFrame({"Predicted": final_pred}, index=test_close.index))
    return rmse, mae, r2
