import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_stock_price(price_data):
    # make copy of data
    data = price_data.copy()

    # error handling
    if data.empty:
        raise ValueError("Price data is empty")

    # get ticker symbol from dataframe and pass to graph title
    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.get_level_values(1)[0]
        data.columns = data.columns.get_level_values(0)
    else:
        ticker = "Stock"

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    close_price = data["Close"]

    # plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, close_price)
    plt.title(f"{ticker} Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price ($)")
    plt.grid(True)
    plt.show()


def plot_trading_volume(price_data):
    # make copy of data
    data = price_data.copy()

    # error handling
    if data.empty:
        raise ValueError("Price data is empty")

    # get ticker symbol from dataframe and pass to graph title
    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.get_level_values(1)[0]
        data.columns = data.columns.get_level_values(0)
    else:
        ticker = "Stock"

    if "Volume" not in data.columns:
        raise ValueError("Data does not contain 'Volume' column")

    # plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Volume"])
    plt.title(f"{ticker} Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume ($)")
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(price_data, predicted_data):
    # make copy of data
    actual = price_data.copy()
    predicted = predicted_data.copy()

    # error handling
    if actual.empty:
        raise ValueError("Price data is empty")
    if predicted.empty:
        raise ValueError("Predicted data is empty")

    # get ticker symbol from dataframe and pass to graph title
    if isinstance(actual.columns, pd.MultiIndex):
        ticker = actual.columns.get_level_values(1)[0]
        actual.columns = actual.columns.get_level_values(0)
    else:
        ticker = "Stock"

    if "Close" not in actual.columns:
        raise ValueError("Data does not contain 'Close' column")
    if "Predicted" not in predicted.columns:
        raise ValueError("Predicted data does not contain 'Predicted' column")

    close_price = actual["Close"]
    predicted_price = predicted["Predicted"]

    # plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, close_price, label="Actual Price")
    plt.plot(predicted.index, predicted_price, label="Predicted Price", linestyle="--")
    plt.title(f"{ticker} Actual vs Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
