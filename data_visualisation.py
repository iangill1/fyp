import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_stock_price(price_data):
    #make copy of data
    data = price_data.copy()

    #error handling
    if data.empty:
        raise ValueError("Price data is empty")

    #get ticker symbol from dataframe and pass to graph title
    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns.get_level_values(1)[0]
        data.columns = data.columns.get_level_values(0)
    else:
        ticker = "Stock"

    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    #plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"])
    plt.title(f"{ticker} Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price ($)")
    plt.grid(True)
    plt.show()
