import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import finnhub
import os
import requests

# load my env file
load_dotenv()

# set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


# retrieve historical stock price data from yfinance api
def price_retrieval(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    # convert to datetime index
    data.index = pd.to_datetime(data.index)
    # set frequency to business day
    data = data.asfreq("B")
    # forward fill missing data (holidays, etc)
    data = data.ffill()
    print(data)
    return data


def price_retrieval_period(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    # convert to datetime index
    data.index = pd.to_datetime(data.index)
    # set frequency to business day
    if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
        data = data.asfreq("B")
        # forward fill missing data (holidays, etc)
        data = data.ffill()
    print(data)
    return data


def price_retrieval_history(ticker, start_date, end_date, interval):
    ticker = yf.Ticker(ticker)
    df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
    # convert to datetime index
    df.index = pd.to_datetime(df.index)
    # set frequency to business day
    df = df.asfreq("B")
    # forward fill missing data (holidays, etc)
    df = df.ffill()
    print(df)
    return df


# retrieve news articles from finnhub api
def finnhub_news_retrieval(ticker, start_date, end_date):
    # get my api key from env file
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    # create finnhub client
    finnhub_client = finnhub.Client(api_key=finnhub_key)
    news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    # convert to data frame
    df = pd.DataFrame(news)
    # convert time to readable datetime
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    df = df[["datetime", "headline", "summary"]]
    print(df)
    return df


# retrieve news articles from alpha vantage api
def alpha_vantage_news_retrieval(ticker, start_date, end_date):
    # convert dates to make them compatible with alpha vantage
    start_date = start_date.replace("-", "") + "T0000"
    end_date = end_date.replace("-", "") + "T2359"
    # get api key from env file
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    # make request to alpha vantage news sentiment endpoint
    url = 'https://www.alphavantage.co/query'
    # set parameters for request
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start_date,
        "time_to": end_date,
        "limit": "1000",
        "apikey": alpha_vantage_key

    }
    # make get request
    response = requests.get(url, params)
    # parse json response
    data = response.json()
    df = pd.DataFrame(data["feed"])
    # if Alpha Vantage returns a "title" column, rename it to "headline" to match finnhub retrieval for easier
    # processing in sentiment analysis
    if "title" in df.columns:
        df.rename(columns={"title": "headline"}, inplace=True)
    # if neither title nor headline exist, create an empty headline column so selection below doesn't fail
    if "headline" not in df.columns:
        df["headline"] = None

    if "time_published" in df.columns:
        df.rename(columns={"time_published": "datetime"}, inplace=True)
    if "datetime" not in df.columns:
        df["datetime"] = None

    # convert time_published to datetime so it can be aligned with stock data
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%dT%H%M%S")
    df = df[["datetime", "headline", "summary"]]
    return df
