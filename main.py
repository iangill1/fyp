from data_retrieval import price_retrieval, price_retrieval_period, alpha_vantage_news_retrieval, finnhub_news_retrieval
from naive_model import naive_model_forecast
from arima_model import arima_model_forecast, plot_autocorrelation, adf_test, determine_differencing, decompose_time_series
from data_visualisation import plot_stock_price, plot_trading_volume, plot_sentiment_over_time
from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_analysis import analyse_sentiment

app = FastAPI(title="Stock Price Forecasting API")


class ForecastRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    interval: str = "1d"


class NewsRetrievalRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    price_data = price_retrieval(req.ticker, req.start_date, req.end_date, req.interval)
    rmse, mae, r2 = arima_model_forecast(price_data)
    return {"Ticker": req.ticker, "ARIMA Model RMSE": rmse, "ARIMA Model MAE": mae, "ARIMA Model r2": r2}


#@app.post("/finnhub_news_retrieval")
#def finnhub_news_retrieval(req: NewsRetrievalRequest):
 #   news_data = finnhub_news_retrieval(req.ticker, req.start_date, req.end_date)
  #  return {"Ticker": req.ticker, "News Data": news_data.to_dict(orient="records")}


def main():
    news = alpha_vantage_news_retrieval("msft", "20100101T0000", "20200101T2359")
    sentiment = analyse_sentiment(news)

    print(sentiment)
    plot_sentiment_over_time(sentiment)
    #news = alpha_vantage_news_retrieval("ttwo", "20151010T0000", "20251231T2359")
    #sentiment = analyse_sentiment(news)
    #print(news)

    """
    data being used
    """
    #msft_upward_trend_data = price_retrieval("MSFT", "2010-01-01", "2020-01-01", "1d")
    #rmse, mae, r2 = arima_model_forecast(msft_upward_trend_data)
    #print(f"MSFT ARIMA Model RMSE: {rmse}, MAE: {mae}, r2: {r2}")
    #barc_downward_trend_data = price_retrieval("BARC.L", "2018-03-01", "2019-07-09", "1d")
    #axp_upward_trend_data = price_retrieval("AXP", "2019-10-01", "2020-04-01", "1d")
    #gme_downward_trend_data = price_retrieval("GME", "2017-05-15", "2019-06-15", "1d")


    #mcdonalds_seasonal_data = price_retrieval("MCD", "2023-03-09", "2025-06-15", "1d")
    #plot_stock_price(mcdonalds_seasonal_data)
    #ccl_seasonal_data = price_retrieval("CCL", "2010-01-01", "2019-12-31", "1mo")
    #plot_stock_price(ccl_seasonal_data)


    #boeing_cyclical_data = price_retrieval("BA", "2023-01-01", "2023-06-30", "1d")
    #rmse1, mae1, r21 = arima_model_forecast(boeing_cyclical_data)
    #print(f"BA ARIMA Model RMSE: {rmse1}, MAE: {mae1}, r2: {r21}")
    #ibm_cyclical_data = price_retrieval("IBM", "2018-02-01", "2022-12-31", "1d")
    #bac_cyclical_data = price_retrieval("BAC", "2017-09-01", "2024-01-15", "1d")
    #ichr_cyclical_data = price_retrieval("ICHR", "2022-07-17", "2025-02-25", "1d")


    #gme_irregular_data = price_retrieval("GME", "2020-11-01", "2021-03-31", "1d")
    #gme_irregular_data1 = price_retrieval("GME", "2018-10-10", "2021-02-15", "1d")
    #ttwo_irregular_data = price_retrieval("ttwo", "2025-10-15", "2026-02-11", "1d")
    #cloudflare_irregular_data = price_retrieval("NET", "2025-10-28", "2025-11-12", "1d")
    #cloudflare_finnhub_news_retrieval = finnhub_news_retrieval("NET", "2025-11-17", "2025-11-18")
    #print(cloudflare_finnhub_news_retrieval)
    #meta_irregular_data = price_retrieval("META", "2018-03-01", "2018-10-31", "1d")
    #plot_stock_price(meta_irregular_data)
    #wfc_irregular_data = price_retrieval("WFC", "2016-01-01", "2016-12-31", "1d")
    #plot_stock_price(wfc_irregular_data)

    """
    # call price retrieval function to retrieve data
    plot_autocorrelation(axp_upward_trend_data)
    #gme_downward_trend_data = price_retrieval("GME", "2017-05-15", "2019-06-15", "1d")
    #walmart_upward_trend_data = price_retrieval("WMT", "2018-06-01", "2020-07-27", "1d")
    #sony_downward_trend_data = price_retrieval("SONY", "2026-01-01", "2026-01-31", "1d")

    #seasonal_data = price_retrieval("MCD", "2011-10-01", "2015-01-01", "1d")
    mcdonalds_seasonal_data = price_retrieval("MCD", "2023-03-09", "2025-06-15", "1d")
    dal_seasonal_data = price_retrieval("DAL", "2005-01-01", "2019-12-31", "1mo")
    """

    """
    ttwo_irregular_data = price_retrieval("ttwo", "2025-10-10", "2025-12-31", "1d")
    ttwo_irregular_data1 = price_retrieval("TTWO", "2025-04-04", "2025-06-25", "1d")
    #msft_irregular_data = price_retrieval_period("MSFT", "1mo", "60m")
    gme_irregular_data = price_retrieval("GME", "2020-11-01", "2021-03-31", "1d")
    """


if __name__ == "__main__":
    main()
