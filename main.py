from data_retrieval import price_retrieval, news_retrieval

def main():
    company_data = price_retrieval("aapl", "2020-01-01", "2023-01-01", "1d")
    company_news = news_retrieval("ttwo", "2025-01-01", "2026-01-01")

    print(company_data.tail())
    print(company_news)  # Print first 5 news articles

if __name__ == "__main__":
    main()