from data_retrieval import price_retrieval, finnhub_news_retrieval, alpha_vantage_news_retrieval
from naive_model import naive_model_forecast
from arima_model import arima_model_forecast
def main():

    # call price retrieval function to retrieve data
    price_data = price_retrieval("ttwo", "2025-01-01", "2026-01-01", "1d")
    news_data = finnhub_news_retrieval("ttwo", "2025-01-01", "2026-01-01")

    #alpha_vantage_news = alpha_vantage_news_retrieval("AAPL", "20260101T0000", "20260109T2359")
    #print(alpha_vantage_news)
    naive_rmse, naive_mae = naive_model_forecast(price_data)
    print("Naive Model RMSE: ", naive_rmse)
    print("Naive Model MAE: ", naive_mae)

    arima_rmse = arima_model_forecast(price_data)
    print("ARIMA model RMSE: ", arima_rmse)


if __name__ == "__main__":
    main()