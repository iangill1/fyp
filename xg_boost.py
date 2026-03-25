from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from data_visualisation import plot_actual_vs_predicted


def xgboost_default_params_forecast(price_data):
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

        model = XGBRegressor(random_state=42).fit(current_train_exog, current_train_close)

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


def xgboost_tuned_forecast(price_data):
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
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [None, 10, 15, 20],
        "subsample": [0.6, 0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42),
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

        model = (XGBRegressor(**best_params, random_state=42))
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