from data_retrieval import price_retrieval, finnhub_news_retrieval, alpha_vantage_news_retrieval
from naive_model import naive_model_forecast
def main():

    # call price retrieval function to retrieve data
    price_data = price_retrieval("ttwo", "2025-01-01", "2026-01-01", "1d")
    news_data = finnhub_news_retrieval("ttwo", "2025-01-01", "2026-01-01")

    alpha_vantage_news = alpha_vantage_news_retrieval("TTWO", "20250101T0000", "20260101T0000")
    print(alpha_vantage_news)
    #rmse, mae = naive_model_forecast(price_data)
    #print("rmse: ", rmse)
    #print("mae: ", mae)

if __name__ == "__main__":
    main()