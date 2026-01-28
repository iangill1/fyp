import matplotlib.pyplot as plt
import seaborn as sns


def plot_stock_price(price_data):
    #make copy of data
    data = price_data.copy()

    #error handling
    if data.empty:
        raise ValueError("Price data is empty")
    if "Close" not in data.columns:
        raise ValueError("Data does not contain 'Close' column")

    #plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'])
    plt.title('S&P 500 Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.grid(True)
    plt.show()
